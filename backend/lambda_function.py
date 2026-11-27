
import base64
import json
import os
import re
import socket
import urllib.error
import urllib.request
from urllib.parse import urlparse

try:
    import boto3
except ImportError:
    boto3 = None


# ============================================================
# CONFIGURATION
# ============================================================

ALLOWED_MARKETPLACES = {
    "amazon.in": "Amazon",
    "www.amazon.in": "Amazon",
    "flipkart.com": "Flipkart",
    "www.flipkart.com": "Flipkart",
}

GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SSM_PARAMETER_NAME = os.environ.get(
    "GEMINI_PARAMETER_NAME",
    "/smart-product-compare/google-api-key"
)

SSM_REGION = os.environ.get("SSM_REGION", "us-east-1")

MAX_BODY_SIZE = 8192
REQUEST_TIMEOUT = 55
MAX_OUTPUT_TOKENS = 5000

VALID_PRIORITIES = {
    "value_for_money",
    "lowest_price",
    "best_features",
    "performance",
    "battery",
    "camera",
    "display",
    "overall"
}

TECHNICAL_CONTENT_MARKERS = [
    "webpack chunk",
    "javascript application/ld+json",
    "window.__next_data__",
    "new relic",
    "newrelic",
    "monitoring payload",
    "javascript bundle",
    "script payload",
    "application/javascript",
    "runtime chunk",
    "minified javascript",
]


# ============================================================
# HTTP RESPONSE
# ============================================================

def response(status_code, payload):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(payload, ensure_ascii=False)
    }


# ============================================================
# REQUEST VALIDATION
# ============================================================

def validate_url(url):
    if not isinstance(url, str) or not url.strip():
        return None

    try:
        parsed = urlparse(url.strip())

        if parsed.scheme != "https":
            return None

        hostname = (parsed.hostname or "").lower()

        if hostname not in ALLOWED_MARKETPLACES:
            return None

        if parsed.username or parsed.password:
            return None

        return {
            "url": url.strip(),
            "marketplace": ALLOWED_MARKETPLACES[hostname]
        }

    except Exception:
        return None


def validate_product_urls(product_urls):
    if not isinstance(product_urls, list):
        return None, "productUrls must be a list."

    if len(product_urls) < 2:
        return None, "At least two product URLs are required."

    if len(product_urls) > 5:
        return None, "A maximum of five products can be compared."

    validated = []

    for url in product_urls:
        item = validate_url(url)

        if not item:
            return None, (
                "Only valid HTTPS Amazon India and Flipkart product URLs "
                "are supported."
            )

        validated.append(item)

    if len(set(item["url"] for item in validated)) != len(validated):
        return None, "Duplicate product URLs are not allowed."

    return validated, None


def validate_budget(budget):
    try:
        value = float(budget)

        if value <= 0:
            return None

        return value

    except (TypeError, ValueError):
        return None


def validate_priority(priority):
    if not isinstance(priority, str):
        return None

    priority = priority.strip().lower()

    if priority not in VALID_PRIORITIES:
        return None

    return priority


# ============================================================
# GEMINI API KEY
# ============================================================

def get_gemini_api_key():
    if boto3 is None:
        raise RuntimeError(
            "boto3 is required to retrieve the Gemini API key from SSM."
        )

    parameter_store = boto3.client(
        "ssm",
        region_name=SSM_REGION
    )

    result = parameter_store.get_parameter(
        Name=SSM_PARAMETER_NAME,
        WithDecryption=True
    )

    return result["Parameter"]["Value"]


# ============================================================
# EVIDENCE VALIDATION
# ============================================================

def is_unavailable(value):
    if value is None:
        return True

    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "unavailable",
            "not available",
            "not found",
            "unknown",
            "n/a",
            "na",
            "null",
            "none",
            "not specified",
            "could not retrieve",
            "failed",
        }

    return False


def looks_like_technical_payload(value):
    if not isinstance(value, str):
        return False

    value = value.lower()

    return any(
        marker in value
        for marker in TECHNICAL_CONTENT_MARKERS
    )


def evidence_is_usable(evidence):
    if not isinstance(evidence, str):
        return False

    evidence = evidence.strip()

    if len(evidence) < 8 or len(evidence) > 1000:
        return False

    if looks_like_technical_payload(evidence):
        return False

    return True


def extract_price_amount(amount, evidence):
    """
    Return a price only when the evidence itself contains an INR currency
    marker and the matching numeric amount. Recover amount from the quote
    when Gemini omits it, but never infer it from summaries or limitations.
    """
    if not evidence_is_usable(evidence):
        return None

    currency_pattern = r"(?:₹|Rs\.?|INR)"
    number_pattern = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"

    match = re.search(
        currency_pattern + r"\s*" + number_pattern
        + r"|" + number_pattern + r"\s*" + currency_pattern,
        evidence,
        re.IGNORECASE
    )

    if not match:
        return None

    quoted_number = match.group(1) or match.group(2)
    if not quoted_number:
        return None

    try:
        evidence_amount = int(float(quoted_number.replace(",", "")))
    except (TypeError, ValueError):
        return None

    if evidence_amount <= 0:
        return None

    if amount is not None:
        try:
            supplied_amount = int(float(
                str(amount).replace(",", "").replace("₹", "").strip()
            ))
        except (TypeError, ValueError):
            supplied_amount = None

        if supplied_amount is not None and supplied_amount != evidence_amount:
            return None

    return evidence_amount


def price_evidence_is_valid(amount, evidence):
    return extract_price_amount(amount, evidence) is not None

def safe_text(value, default="Unavailable"):
    if is_unavailable(value):
        return default

    if not isinstance(value, str):
        return default

    value = value.strip()

    if looks_like_technical_payload(value):
        return default

    return value[:500]


def safe_text_list(items):
    if not isinstance(items, list):
        return []

    result = []

    for item in items:
        if isinstance(item, str):
            cleaned = safe_text(item, default="")
            if cleaned:
                result.append(cleaned)

        elif isinstance(item, dict):
            cleaned = safe_text(
                item.get("text") or item.get("description"),
                default=""
            )
            if cleaned:
                result.append(cleaned)

    return result[:5]


# ============================================================
# VARIANT VALIDATION
# ============================================================

def validate_variant(product):
    model_match = product.get("model_match")
    storage_match = product.get("storage_match")
    color_match = product.get("color_match")
    legacy_match = product.get("variant_match")

    # Compatibility with existing test fixtures and older responses.
    if (
        legacy_match is True
        and model_match is None
        and storage_match is None
    ):
        return "exact"

    if model_match is False or storage_match is False:
        return "mismatch"

    if model_match is True and storage_match is True:
        if color_match is False:
            return "partial"

        if color_match is True:
            return "exact"

        return "partial"

    if legacy_match is True:
        return "exact"

    return "unknown"


def comparison_eligible(product):
    """
    A product can participate in a comparison when:
    - identity is supported;
    - at least two specifications have evidence;
    - model and storage match;
    - colour may be exact or different.

    Legacy exact-match responses remain supported.
    """
    if product.get("retrieval_status") not in ("verified", "partial"):
        return False

    specs = product.get("key_specifications", [])

    if len(specs) < 2:
        return False

    if product.get("variant_status") == "exact":
        return True

    return (
        product.get("model_match") is True
        and product.get("storage_match") is True
        and product.get("variant_status") == "partial"
    )


# ============================================================
# TECHNICAL PAYLOAD DETECTION
# ============================================================

def product_has_technical_payload(product):
    if looks_like_technical_payload(
        product.get("retrieval_notes", "")
    ):
        return True

    if looks_like_technical_payload(
        product.get("product_name", "")
    ):
        return True

    for spec in product.get("key_specifications", []):
        if not isinstance(spec, dict):
            continue

        for field in ("name", "value", "evidence"):
            if looks_like_technical_payload(spec.get(field, "")):
                return True

    return False


# ============================================================
# VARIANT DISPLAY HELPERS
# ============================================================

def variant_display(value):
    if isinstance(value, str):
        return safe_text(value, default="Unknown")

    if isinstance(value, dict):
        parts = []

        for key in ("color", "colour", "storage", "ram", "model"):
            item = value.get(key)

            if isinstance(item, str) and not is_unavailable(item):
                parts.append(f"{key.title()}: {item.strip()}")

        return ", ".join(parts) if parts else "Unknown"

    return "Unknown"


def build_variant_warning(variant_status, requested, retrieved):
    if variant_status == "exact":
        return "Retrieved variant matches the requested variant."

    if variant_status == "partial":
        return (
            "The retrieved product has a different or unconfirmed colour "
            "variant. Its specifications and price belong to the retrieved "
            "variant, not necessarily the requested colour."
        )

    if variant_status == "mismatch":
        return (
            "The retrieved model or storage does not match the requested "
            "product. Do not treat its price as the requested product's price."
        )

    return "The requested variant could not be fully verified."


# ============================================================
# NORMALIZE COMPARISON
# ============================================================

def normalize_comparison(comparison, submitted_products, budget=None):
    if not isinstance(comparison, dict):
        comparison = {}

    ai_products = comparison.get("products", [])

    if not isinstance(ai_products, list):
        ai_products = []

    normalized_products = []

    for index, submitted in enumerate(submitted_products):
        product = (
            ai_products[index]
            if index < len(ai_products)
            and isinstance(ai_products[index], dict)
            else {}
        )

        marketplace = submitted.get("marketplace", "Unknown")
        product_url = submitted.get("url", "")

        # Reject script payloads before trusting any AI-inferred details.
        if product_has_technical_payload(product):
            normalized_products.append({
                "product_name": "Unavailable",
                "brand": "Unavailable",
                "marketplace": marketplace,
                "url": product_url,
                "retrieval_status": "failed",
                "retrieval_notes": (
                    "The retrieved page contained technical or script "
                    "content instead of usable product information."
                ),
                "variant_match": False,
                "variant_status": "unknown",
                "model_match": None,
                "storage_match": None,
                "color_match": None,
                "requested_variant": "Unknown",
                "retrieved_variant": "Unknown",
                "variant_warning": "Product variant could not be verified.",
                "key_specifications": [],
                "price": {
                    "amount": None,
                    "currency": "INR",
                    "status": "not_found",
                    "evidence": None,
                    "variant": "Unknown",
                    "budget_status": "unknown"
                },
                "budget_status": "unknown",
                "availability": "Unknown",
                "strengths": [],
                "tradeoffs": []
            })
            continue

        product_name = safe_text(product.get("product_name"))
        brand = safe_text(product.get("brand"))

        identity_valid = evidence_is_usable(
            product.get("identity_evidence")
        )

        variant_status = validate_variant(product)

        # ----------------------------------------------------
        # SPECIFICATIONS
        # ----------------------------------------------------

        specifications = []
        raw_specs = product.get("key_specifications", [])

        if isinstance(raw_specs, list):
            for spec in raw_specs:
                if not isinstance(spec, dict):
                    continue

                name = spec.get("name")
                value = spec.get("value")
                evidence = spec.get("evidence")

                if is_unavailable(name) or is_unavailable(value):
                    continue

                if not evidence_is_usable(evidence):
                    continue

                if (
                    looks_like_technical_payload(name)
                    or looks_like_technical_payload(value)
                ):
                    continue

                specifications.append({
                    "name": str(name).strip()[:100],
                    "value": str(value).strip()[:200],
                    "evidence": evidence.strip()[:500]
                })

        # ----------------------------------------------------
        # VARIANT DETAILS
        # ----------------------------------------------------

        requested_variant = variant_display(
            product.get("requested_variant")
        )

        retrieved_variant = variant_display(
            product.get("retrieved_variant")
        )

        if requested_variant == "Unknown":
            requested_variant = variant_display(
                product.get("requested_color")
            )

        if retrieved_variant == "Unknown":
            retrieved_variant = variant_display(
                product.get("retrieved_color")
            )

        variant_warning = build_variant_warning(
            variant_status,
            requested_variant,
            retrieved_variant
        )

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        raw_price = product.get("price", {})

        if not isinstance(raw_price, dict):
            raw_price = {}

        price_amount = raw_price.get("amount")
        price_evidence = raw_price.get("evidence")

        # The quoted page evidence is authoritative. Recover the number
        # from it when Gemini returns amount=null, but never infer a price
        # from summaries, feature rows, or free-form limitations.
        price_amount = extract_price_amount(
            price_amount,
            price_evidence
        )
        valid_price = price_amount is not None

        if price_amount is None:
            budget_status = "unknown"
        elif budget is None:
            budget_status = "unknown"
        elif price_amount <= budget:
            budget_status = "within_budget"
        else:
            budget_status = "over_budget"

        price = {
            "amount": price_amount,
            "currency": "INR",
            "status": "verified" if valid_price else "not_found",
            "evidence": (
                price_evidence.strip()[:500]
                if valid_price
                else None
            ),
            "variant": retrieved_variant,
            "budget_status": budget_status
        }

        # ----------------------------------------------------
        # RETRIEVAL STATUS
        # ----------------------------------------------------

        if variant_status == "mismatch":
            retrieval_status = "partial"

        elif (
            identity_valid
            and len(specifications) >= 2
            and variant_status == "exact"
        ):
            retrieval_status = "verified"

        elif identity_valid or specifications or valid_price:
            retrieval_status = "partial"

        else:
            retrieval_status = "failed"

        if retrieval_status == "failed":
            product_name = "Unavailable"
            brand = "Unavailable"
            specifications = []

            price = {
                "amount": None,
                "currency": "INR",
                "status": "not_found",
                "evidence": None,
                "variant": "Unknown",
                "budget_status": "unknown"
            }

            budget_status = "unknown"

        # ----------------------------------------------------
        # AVAILABILITY
        # ----------------------------------------------------

        availability = safe_text(
            product.get("availability"),
            default="Unknown"
        )

        # Avoid passing unsupported AI-generated claims to the UI.
        strengths = []
        tradeoffs = []

        normalized_products.append({
            "product_name": product_name,
            "brand": brand,
            "marketplace": marketplace,
            "url": product_url,
            "retrieval_status": retrieval_status,
            "retrieval_notes": safe_text(
                product.get("retrieval_notes"),
                default=""
            ),
            "variant_match": variant_status == "exact",
            "variant_status": variant_status,
            "model_match": product.get("model_match"),
            "storage_match": product.get("storage_match"),
            "color_match": product.get("color_match"),
            "requested_variant": requested_variant,
            "retrieved_variant": retrieved_variant,
            "variant_warning": variant_warning,
            "key_specifications": specifications,
            "price": price,
            "budget_status": budget_status,
            "availability": availability,
            "strengths": strengths,
            "tradeoffs": tradeoffs
        })

    # ========================================================
    # COMPARISON ELIGIBILITY
    # ========================================================

    eligible_products = [
        product
        for product in normalized_products
        if comparison_eligible(product)
    ]

    raw_comparison = comparison.get("comparison", {})

    if not isinstance(raw_comparison, dict):
        raw_comparison = {}

    if len(eligible_products) < 2:
        feature_comparison = []

        recommendations = {
            "best_overall": "insufficient data",
            "best_value": "insufficient data",
            "best_features": "insufficient data",
            "reason": (
                "At least two products must have sufficient comparable "
                "information before recommendations can be generated."
            )
        }

    else:
        raw_features = raw_comparison.get(
            "feature_comparison",
            []
        )

        feature_comparison = (
            raw_features if isinstance(raw_features, list) else []
        )

        raw_recommendations = comparison.get("recommendations", {})

        if not isinstance(raw_recommendations, dict):
            raw_recommendations = {}

        allowed_names = {
            product["product_name"]
            for product in eligible_products
        }

        def safe_recommendation(value):
            if isinstance(value, str) and value in allowed_names:
                return value

            return "insufficient data"

        recommendations = {
            "best_overall": safe_recommendation(
                raw_recommendations.get("best_overall")
            ),
            "best_value": safe_recommendation(
                raw_recommendations.get("best_value")
            ),
            "best_features": safe_recommendation(
                raw_recommendations.get("best_features")
            )
        }

    # ========================================================
    # PRICE COMPARISON
    # ========================================================

    priced_products = [
        product
        for product in eligible_products
        if product["price"]["amount"] is not None
    ]

    price_comparison = {
        "lowest_price": "insufficient data",
        "highest_price": "insufficient data",
        "price_difference": None
    }

    if len(priced_products) >= 2:
        lowest = min(
            priced_products,
            key=lambda item: item["price"]["amount"]
        )

        highest = max(
            priced_products,
            key=lambda item: item["price"]["amount"]
        )

        price_comparison = {
            "lowest_price": lowest["product_name"],
            "highest_price": highest["product_name"],
            "price_difference": (
                highest["price"]["amount"]
                - lowest["price"]["amount"]
            )
        }

    # ========================================================
    # DATA LIMITATIONS
    # ========================================================

    limitations = comparison.get("data_limitations", [])

    if not isinstance(limitations, list):
        limitations = []

    limitations = safe_text_list(limitations)

    for product in normalized_products:
        if product["budget_status"] == "over_budget":
            message = (
                f'{product["product_name"]} exceeds the budget '
                f'of ₹{budget:,.0f}.'
            )
            if message not in limitations:
                limitations.append(message)

        if product["variant_status"] == "partial":
            message = (
                f'{product["product_name"]}: retrieved variant may differ '
                "from the requested colour. Its price belongs to the "
                "retrieved variant."
            )
            if message not in limitations:
                limitations.append(message)

        if product["retrieval_status"] == "failed":
            message = (
                f'{product["marketplace"]} product information '
                "could not be reliably retrieved."
            )
            if message not in limitations:
                limitations.append(message)

    return {
        "products": normalized_products,
        "comparison": {
            "feature_comparison": feature_comparison,
            "price_comparison": price_comparison
        },
        "recommendations": recommendations,
        "data_limitations": limitations
    }


# ============================================================
# GEMINI INPUT
# ============================================================

def prepare_ai_input(event):
    category = event.get("category")

    if not isinstance(category, str) or not category.strip():
        return None, "A product category is required."

    submitted_products, error = validate_product_urls(
        event.get("productUrls")
    )

    if error:
        return None, error

    budget = validate_budget(event.get("budget"))

    if budget is None:
        return None, "A valid budget greater than zero is required."

    priority = validate_priority(
        event.get("priority", "value_for_money")
    )

    if priority is None:
        return None, "The selected comparison priority is invalid."

    return {
        "category": category.strip()[:100],
        "budget": budget,
        "priority": priority,
        "products": submitted_products
    }, None


# ============================================================
# GEMINI PROMPT
# ============================================================

def build_gemini_prompt(data, use_search=False):
    products_json = json.dumps(
        data["products"],
        ensure_ascii=False
    )

    search_instruction = ""

    if use_search:
        search_instruction = (
            "Use Google Search only as supplementary research. "
            "Do not treat snippets as proof of a current price."
        )

    return f"""
You are a careful product research and comparison assistant.

Compare these products using the submitted marketplace URLs.

Category: {data["category"]}
User budget: INR {data["budget"]}
User priority: {data["priority"]}

Submitted URLs:
{products_json}

Use URL Context to inspect the actual product pages.
{search_instruction}

EVIDENCE RULES:

1. Never invent product names, brands, specifications, prices,
   availability, or variants.

2. Every product identity needs identity_evidence.

3. Every specification needs its own evidence from the page.

4. Omit specifications without evidence.

5. For each price, copy a short, exact quote from the product page
   into price.evidence. The quote MUST contain both the INR currency
   marker (₹, Rs., or INR) and the complete price digits.

6. Put the price as a JSON number in price.amount (digits only, no
   commas or currency symbol). Example: amount=89999 and evidence=
   "Special price ₹89,999". Do not put a price from a search snippet,
   comparison text, or memory into the price field.

7. If the page does not show a verifiable current price, use
   amount=null, status="not_found", and evidence=null. Never guess.

7. Identify the requested model and storage and compare them with
   the retrieved product.

8. A different colour is a PARTIAL variant match if model and storage
   match. Do not reject it solely because the colour differs.

9. Clearly return requested_variant and retrieved_variant.
   Include colour and storage where the page supports them.

10. A different model or storage is a mismatch. Never use its price
    as the price of the requested product.

11. A product is verified when identity evidence exists, at least
    two specifications have evidence, and its requested variant
    is an exact match.

12. A product with matching model and storage but different colour
    is partial. Its price belongs to the retrieved colour variant.

13. Compare eligible products when at least two products have
    matching model and storage, sufficient evidence, and usable
    specifications. Partial colour matches may participate, but
    clearly disclose the colour difference.

14. Never recommend a failed or mismatched product.

15. Never put unsupported prices in summaries or recommendations.

16. Budget does not determine whether product data is verified.
    Report over-budget products separately.

17. If a page contains only scripts, JavaScript bundles, monitoring
    payloads, or unrelated technical content, mark it failed.

Return ONLY valid JSON, without Markdown fences.

Use this structure:

{{
  "products": [
    {{
      "product_name": "Product name",
      "brand": "Brand",
      "identity_evidence": "Evidence from page",
      "retrieval_status": "verified, partial, or failed",
      "retrieval_notes": "Explain retrieval and variant details",
      "model_match": true,
      "storage_match": true,
      "color_match": true,
      "variant_match": true,
      "requested_variant": {{
        "color": "Requested colour",
        "storage": "Requested storage"
      }},
      "retrieved_variant": {{
        "color": "Retrieved colour",
        "storage": "Retrieved storage"
      }},
      "key_specifications": [
        {{
          "name": "Specification",
          "value": "Value",
          "evidence": "Page evidence"
        }}
      ],
      "price": {{
        "amount": null,
        "status": "verified or not_found",
        "evidence": null
      }},
      "availability": "Available, unavailable, or Unknown",
      "strengths": [],
      "tradeoffs": []
    }}
  ],
  "comparison": {{
    "feature_comparison": []
  }},
  "recommendations": {{
    "best_overall": "Product name or insufficient data",
    "best_value": "Product name or insufficient data",
    "best_features": "Product name or insufficient data"
  }},
  "data_limitations": []
}}
"""


# ============================================================
# GEMINI REQUEST
# ============================================================


def parse_gemini_json(output_text):
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("Gemini returned an empty response.")

    text = output_text.strip()

    # Remove Markdown code fences, if present.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"\s*```$", "", text)

    # Extract the JSON object if Gemini adds introductory text.
    start = text.find("{")

    if start == -1:
        raise ValueError("Gemini response contains no JSON object.")

    text = text[start:]

    # Repair common formatting mistakes.
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r":\s*,", ": null,", text)
    text = re.sub(r":\s*}", ": null}", text)

    try:
        result, _ = json.JSONDecoder().raw_decode(text)

        if not isinstance(result, dict):
            raise ValueError("Gemini JSON response is not an object.")

        return result

    except json.JSONDecodeError as error:
        # Log a small excerpt for diagnosis, never the API key.
        excerpt = text[
            max(0, error.pos - 150):
            min(len(text), error.pos + 150)
        ]

        print(
            "Gemini JSON parsing failed:",
            f"line={error.lineno}",
            f"column={error.colno}",
            f"excerpt={excerpt!r}"
        )

        raise ValueError(
            f"Gemini returned malformed JSON at "
            f"line {error.lineno}, column {error.colno}."
        ) from error


def call_gemini_once(data, api_key, use_search=False):
    prompt = build_gemini_prompt(data, use_search)

    tools = [{"url_context": {}}]

    if use_search:
        tools.append({"google_search": {}})

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "Return only valid JSON. Ground product facts in "
                    "the supplied pages. Never invent information."
                )
            }]
        },
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "tools": tools,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json"
        }
    }

    request = urllib.request.Request(
        f"{GEMINI_ENDPOINT}?key={api_key}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT
    ) as result:
        response_data = json.loads(
            result.read().decode("utf-8")
        )

    candidates = response_data.get("candidates", [])

    if not candidates:
        raise ValueError("Gemini returned no response candidates.")

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    output_text = "\n".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ).strip()

    if not output_text:
        raise ValueError("Gemini returned an empty response.")

    output_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        output_text,
        flags=re.IGNORECASE
    )

    output_text = re.sub(r"\s*```$", "", output_text)

    return parse_gemini_json(output_text)

# ============================================================
# GEMINI FALLBACK
# ============================================================

def needs_search_fallback(comparison):
    products = comparison.get("products", [])

    if not isinstance(products, list) or len(products) < 2:
        return True

    for product in products:
        if not isinstance(product, dict):
            return True

        if not evidence_is_usable(
            product.get("identity_evidence")
        ):
            return True

        specs = product.get("key_specifications", [])

        if not isinstance(specs, list) or not specs:
            return True

    return False


def call_gemini(data):
    api_key = get_gemini_api_key()

    first_comparison = call_gemini_once(
        data,
        api_key,
        use_search=False
    )

    metadata = {
        "model": GEMINI_MODEL,
        "urlContextUsed": True,
        "googleSearchFallbackUsed": False
    }

    if not needs_search_fallback(first_comparison):
        return first_comparison, metadata

    try:
        fallback_comparison = call_gemini_once(
            data,
            api_key,
            use_search=True
        )

        fallback_products = fallback_comparison.get("products", [])
        original_products = first_comparison.get("products", [])

        if (
            isinstance(fallback_products, list)
            and len(fallback_products) >= len(original_products)
        ):
            metadata["googleSearchFallbackUsed"] = True
            return fallback_comparison, metadata

        metadata["fallbackNote"] = (
            "Search fallback did not improve the retrieved result."
        )

    except Exception as error:
        metadata["fallbackError"] = str(error)[:500]

    return first_comparison, metadata


# ============================================================
# AI SUMMARY HANDLER
# ============================================================

def handle_ai_summary(event):
    data, validation_error = prepare_ai_input(event)

    if validation_error:
        return response(400, {
            "status": "validation_error",
            "message": validation_error
        })

    try:
        comparison, metadata = call_gemini(data)

        normalized = normalize_comparison(
            comparison,
            data["products"],
            budget=data["budget"]
        )

        data_retrieved = any(
            product["retrieval_status"] in ("verified", "partial")
            for product in normalized["products"]
        )

        return response(200, {
            "status": "comparison_generated",
            "model": metadata.get("model", GEMINI_MODEL),
            "dataRetrieved": data_retrieved,
            "metadata": metadata,
            "category": data["category"],
            "budget": data["budget"],
            "priority": data["priority"],
            **normalized
        })

    except urllib.error.HTTPError as error:
        error_body = ""

        try:
            error_body = error.read().decode("utf-8")[:1000]
        except Exception:
            pass

        return response(502, {
            "status": "gemini_api_error",
            "message": "The Gemini API request failed.",
            "details": error_body or str(error)
        })

    except (TimeoutError, socket.timeout):
        return response(504, {
            "status": "timeout",
            "message": "The product comparison request timed out."
        })

    except urllib.error.URLError as error:
        return response(502, {
            "status": "network_error",
            "message": "Unable to connect to the Gemini API.",
            "details": str(error.reason)[:500]
        })

    except Exception as error:
        print("AI summary error:", str(error))

        return response(500, {
            "status": "internal_error",
            "message": "Unable to generate the product comparison.",
            "details": str(error)[:500]
        })


# ============================================================
# LEGACY URL VALIDATION HANDLER
# ============================================================

def handle_legacy_validation(event):
    submitted, error = validate_product_urls(
        event.get("productUrls")
    )

    if error:
        return response(400, {
            "status": "correction_required",
            "message": error
        })

    budget = validate_budget(event.get("budget"))

    if budget is None:
        return response(400, {
            "status": "correction_required",
            "message": "A valid budget greater than zero is required."
        })

    priority = validate_priority(
        event.get("priority", "value_for_money")
    )

    if priority is None:
        return response(400, {
            "status": "correction_required",
            "message": "The selected priority is invalid."
        })

    return response(200, {
        "status": "validation_success",
        "totalProducts": len(submitted),
        "products": submitted,
        "budget": budget,
        "priority": priority
    })


# ============================================================
# AWS LAMBDA ENTRY POINT
# ============================================================

def lambda_handler(event, context):
    if not isinstance(event, dict):
        return response(400, {
            "status": "validation_error",
            "message": "Invalid request."
        })

    request_context = event.get("requestContext", {})
    method = None

    if isinstance(request_context, dict):
        http_context = request_context.get("http", {})

        if isinstance(http_context, dict):
            method = http_context.get("method")

    if method == "OPTIONS":
        return response(200, {"status": "success"})

    if method and method != "POST":
        return response(405, {
            "status": "method_not_allowed",
            "message": "Only POST requests are supported."
        })

    # Preserve the Function URL envelope while parsing its body.
    if "body" in event:
        raw_body = event.get("body")
        is_base64 = event.get("isBase64Encoded", False)

        if isinstance(raw_body, str):
            try:
                if is_base64:
                    raw_body = base64.b64decode(
                        raw_body
                    ).decode("utf-8")

                if len(raw_body.encode("utf-8")) > MAX_BODY_SIZE:
                    return response(413, {
                        "status": "request_too_large",
                        "message": "Request body exceeds the allowed size."
                    })

                event = json.loads(raw_body)

            except (ValueError, UnicodeDecodeError):
                return response(400, {
                    "status": "validation_error",
                    "message": "Request body must contain valid JSON."
                })

        elif isinstance(raw_body, dict):
            event = raw_body

        else:
            return response(400, {
                "status": "validation_error",
                "message": "Invalid request body."
            })

    if not isinstance(event, dict):
        return response(400, {
            "status": "validation_error",
            "message": "Invalid request payload."
        })

    if event.get("action") == "ai_summary":
        return handle_ai_summary(event)

    return handle_legacy_validation(event)
