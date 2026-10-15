
import ipaddress
import json
import socket
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urljoin

import dns.resolver
import dns.reversename
import requests
import whois


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "model_exports" / "phishguard_config.json"

DNS_TIMEOUT = 3
HTTP_TIMEOUT = 5
MAX_REDIRECTS = 10

with open(CONFIG_PATH, encoding="utf-8") as file:
    CONFIG = json.load(file)

FEATURE_COLUMNS = CONFIG["feature_columns"]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd",
    "ow.ly", "buff.ly", "cutt.ly", "rebrand.ly",
    "shorturl.at", "rb.gy", "tiny.cc",
}


def _resolver():
    resolver = dns.resolver.Resolver()
    resolver.timeout = DNS_TIMEOUT
    resolver.lifetime = DNS_TIMEOUT
    return resolver


def _normalise_url(url):
    """Normalise input and allow only HTTP or HTTPS URLs."""
    url = url.strip()

    if not url:
        raise ValueError("Please enter a URL.")

    if any(ord(char) < 32 for char in url):
        raise ValueError("The URL contains invalid control characters.")

    candidate = url if "://" in url else "https://" + url
    parsed = urlparse(candidate)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    if not parsed.hostname:
        raise ValueError("Please enter a URL with a valid hostname.")

    try:
        parsed.port
    except ValueError as exc:
        raise ValueError("The URL contains an invalid port.") from exc

    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded usernames or passwords are not supported.")

    return candidate, parsed


def _public_ip_addresses(hostname):
    """
    Resolve a hostname and reject private, loopback, link-local,
    reserved, multicast, and otherwise non-global IP addresses.
    """
    try:
        literal_ip = ipaddress.ip_address(hostname)
        addresses = [literal_ip]
    except ValueError:
        addresses = []

        for record_type in ("A", "AAAA"):
            try:
                answers = _resolver().resolve(
                    hostname, record_type, raise_on_no_answer=False
                )
                if answers.rrset:
                    addresses.extend(
                        ipaddress.ip_address(answer.address)
                        for answer in answers
                    )
            except Exception:
                continue

    unique_addresses = list(dict.fromkeys(addresses))

    if not unique_addresses:
        raise ValueError(
            f"Could not resolve the hostname '{hostname}'. "
            "Check the URL or try again later."
        )

    if any(not address.is_global for address in unique_addresses):
        raise ValueError(
            "This URL resolves to a non-public IP address. "
            "For safety, PhishGuard will not access it."
        )

    return unique_addresses


def _safe_request(url):
    """
    Make a bounded HTTP request, checking every redirect destination
    before connecting. Returns redirect count and elapsed time.
    """
    current_url = url
    redirects = 0
    started = time.perf_counter()

    headers = {
        "User-Agent": "PhishGuard-Research/1.0",
        "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    }

    with requests.Session() as session:
        session.trust_env = False

        while True:
            _, parsed = _normalise_url(current_url)
            _public_ip_addresses(parsed.hostname)

            try:
                response = session.get(
                    current_url,
                    headers=headers,
                    timeout=HTTP_TIMEOUT,
                    allow_redirects=False,
                    stream=True,
                    verify=True,
                )
            except requests.RequestException:
                raise

            try:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")

                    if not location:
                        elapsed = time.perf_counter() - started
                        return redirects, elapsed

                    if redirects >= MAX_REDIRECTS:
                        raise ValueError(
                            "The URL exceeded the redirect limit."
                        )

                    next_url = urljoin(current_url, location)
                    _, next_parsed = _normalise_url(next_url)

                    # Check the next host before requesting it.
                    _public_ip_addresses(next_parsed.hostname)

                    current_url = next_url
                    redirects += 1
                    continue

                elapsed = time.perf_counter() - started
                return redirects, elapsed
            finally:
                response.close()


def _dns_records(hostname, record_type):
    """Return DNS answers, or an empty list if lookup fails."""
    try:
        answer = _resolver().resolve(
            hostname, record_type, raise_on_no_answer=False
        )
        if answer.rrset is None:
            return []
        return list(answer)
    except Exception:
        return []


def _domain_whois_dates(hostname):
    """Return WHOIS creation and expiration dates when available."""
    try:
        result = whois.whois(hostname)
    except Exception:
        return None, None

    creation = getattr(result, "creation_date", None)
    expiration = getattr(result, "expiration_date", None)

    # Some WHOIS servers return a list of dates.
    if isinstance(creation, list):
        creation = min((d for d in creation if d), default=None)

    if isinstance(expiration, list):
        expiration = max((d for d in expiration if d), default=None)

    def to_utc(value):
        if not value:
            return None
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    return to_utc(creation), to_utc(expiration)


def _domain_age_features(hostname):
    """Days since domain creation and days until expiration."""
    created, expires = _domain_whois_dates(hostname)
    now = datetime.now(timezone.utc)

    activation_days = (
        (now - created).total_seconds() / 86400
        if created else None
    )
    expiration_days = (
        (expires - now).total_seconds() / 86400
        if expires else None
    )

    return activation_days, expiration_days


def _spf_present(hostname):
    """Check the domain and its parent domains for a TXT SPF record."""
    labels = hostname.split(".")

    # Check the full hostname, then parent domains.
    for index in range(len(labels) - 1):
        domain = ".".join(labels[index:])

        for record in _dns_records(domain, "TXT"):
            try:
                text = b"".join(record.strings).decode(
                    "utf-8", errors="ignore"
                )
            except Exception:
                text = record.to_text().strip('"')

            if text.lower().startswith("v=spf1"):
                return 1

    return 0




def _asn_for_ip(ip_address):
    """Look up the originating ASN for an IPv4 or IPv6 address."""
    try:
        ip = ipaddress.ip_address(str(ip_address))

        if isinstance(ip, ipaddress.IPv4Address):
            reversed_ip = ".".join(reversed(str(ip).split(".")))
            query_name = f"{reversed_ip}.origin.asn.cymru.com"
        else:
            # The reverse pointer is already in reversed-nibble order.
            reversed_ip = ip.reverse_pointer.removesuffix(".ip6.arpa.")
            query_name = f"{reversed_ip}.origin6.asn.cymru.com"

        answers = _resolver().resolve(query_name, "TXT")

        for answer in answers:
            text = b"".join(answer.strings).decode(
                "utf-8", errors="ignore"
            )
            asn = text.split("|", 1)[0].strip()
            if asn.isdigit():
                return int(asn)

    except Exception as exc:
        print(
            f"ASN lookup failed for {ip_address}: "
            f"{type(exc).__name__}: {exc}"
        )

    return None




def _ttl_for_hostname(hostname):
    """Return the lowest TTL among successful A/AAAA DNS answers."""
    ttls = []

    for record_type in ("A", "AAAA"):
        try:
            answer = _resolver().resolve(
                hostname, record_type, raise_on_no_answer=False
            )
            if answer.rrset is not None:
                ttls.append(int(answer.rrset.ttl))
        except Exception:
            continue

    return min(ttls) if ttls else None


def _valid_tls_certificate(hostname):
    """Return 1 if a trusted TLS certificate is presented, otherwise 0."""
    context = ssl.create_default_context()

    try:
        with socket.create_connection(
            (hostname, 443), timeout=HTTP_TIMEOUT
        ) as raw_socket:
            with context.wrap_socket(
                raw_socket, server_hostname=hostname
            ):
                return 1
    except (OSError, ssl.SSLError, ValueError):
        return 0


def _url_based_features(url, parsed):
    """Calculate lexical features from the URL components."""
    hostname = parsed.hostname.lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""

    # Dataset component convention: directory excludes the final path
    # component, while file is the final component.
    if "/" in path:
        directory = path.rsplit("/", 1)[0]
        filename = path.rsplit("/", 1)[-1]
    else:
        directory = ""
        filename = path

    domain_parts = hostname.split(".")
    tld = domain_parts[-1] if len(domain_parts) > 1 else ""

    try:
        ipaddress.ip_address(hostname)
        domain_is_ip = 1
    except ValueError:
        domain_is_ip = 0

    query_params = parse_qs(query, keep_blank_values=True)

    return {
        "qty_slash_url": url.count("/"),
        "directory_length": len(directory),
        "length_url": len(url),
        "qty_dot_domain": hostname.count("."),
        "qty_slash_directory": directory.count("/"),
        "domain_length": len(hostname),
        "file_length": len(filename),
        "qty_hyphen_directory": directory.count("-"),
        "qty_vowels_domain": sum(hostname.count(v) for v in "aeiou"),
        "qty_dot_url": url.count("."),
        "qty_dot_params": query.count("."),
        "qty_tld_url": len(tld),
        "qty_at_url": url.count("@"),
        "qty_hyphen_domain": hostname.count("-"),
        "qty_hyphen_url": url.count("-"),
        "params_length": len(query),
        "qty_params": len(query_params),
        "url_shortened": int(
            hostname in SHORTENER_DOMAINS
            or any(hostname.endswith("." + d) for d in SHORTENER_DOMAINS)
        ),
        "qty_dot_directory": directory.count("."),
        "qty_hyphen_file": filename.count("-"),
        "qty_dot_file": filename.count("."),
        "qty_equal_directory": directory.count("="),
        "qty_underline_directory": directory.count("_"),
        "qty_underline_url": url.count("_"),
        "qty_percent_url": url.count("%"),
        "qty_underline_file": filename.count("_"),
        "qty_percent_file": filename.count("%"),
        "qty_equal_url": url.count("="),
        "domain_in_ip": domain_is_ip,
        "qty_percent_directory": directory.count("%"),
        "qty_equal_params": query.count("="),
        "qty_plus_url": url.count("+"),
        "qty_exclamation_directory": directory.count("!"),
        "qty_slash_params": query.count("/"),
        "qty_percent_params": query.count("%"),
        "qty_underline_params": query.count("_"),
        "qty_comma_file": filename.count(","),
        "tld_present_params": int(bool(tld and tld in query)),
        "qty_tilde_directory": directory.count("~"),
    }


def extract_url_features(url):
    """
    Extract all 50 model features in the trained feature order.

    Returns a dict with numeric values where available. Any unavailable
    external lookup stays None and is reported by the caller; no unknown
    values are silently replaced with fabricated numbers.
    """
    candidate, parsed = _normalise_url(url)
    hostname = parsed.hostname.lower().rstrip(".")

    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("Localhost URLs are not supported.")

    # Resolve and validate the initial host before any outbound HTTP call.
    addresses = _public_ip_addresses(hostname)

    values = _url_based_features(candidate, parsed)

    # External lookup features.
    activation_days, expiration_days = _domain_age_features(hostname)
    values["time_domain_activation"] = activation_days
    values["time_domain_expiration"] = expiration_days

    values["ttl_hostname"] = _ttl_for_hostname(hostname)
    values["qty_ip_resolved"] = len(addresses)

    asn = None
    for address in addresses:
        asn = _asn_for_ip(address)
        if asn is not None:
            break
    values["asn_ip"] = asn

    values["qty_nameservers"] = len(_dns_records(hostname, "NS"))
    values["qty_mx_servers"] = len(_dns_records(hostname, "MX"))
    values["domain_spf"] = _spf_present(hostname)
    values["tls_ssl_certificate"] = _valid_tls_certificate(hostname)

    try:
        redirects, response_time = _safe_request(candidate)
        values["qty_redirects"] = redirects
        values["time_response"] = response_time
    except (requests.RequestException, ValueError):
        values["qty_redirects"] = None
        values["time_response"] = None

    # Match the exact column order expected by the saved model.
    ordered = {name: values.get(name) for name in FEATURE_COLUMNS}

    if len(ordered) != 50:
        raise RuntimeError(
            f"Expected 50 features but assembled {len(ordered)}."
        )

    return ordered


def get_missing_features(features):
    """Return feature names whose values could not be extracted."""
    return [name for name, value in features.items() if value is None]
