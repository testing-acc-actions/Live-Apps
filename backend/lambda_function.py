
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
    "webpack",
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
        normalized = value.strip().lower()

        return normalized in {
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

    text = value.lower()

    return any(
        marker in text
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


def price_evidence_is_valid(amount, evidence):
    if amount is None or not evidence_is_usable(evidence):
        return False

    try:
        numeric_amount = int(float(amount))
    except (TypeError, ValueError):
        return False

    if numeric_amount <= 0:
        return False

    normalized = evidence.replace(",", "")
    normalized = normalized.replace(" ", "")

    amount_pattern = rf"(?<!\d){numeric_amount}(?!\d)"

    has_amount = re.search(
        amount_pattern,
        normalized
    ) is not None

    has_currency = re.search(
        r"₹|Rs\.?|INR",
        evidence,
        re.IGNORECASE
    ) is not None

    return has_amount and has_currency


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
            text = item.get("text") or item.get("description")
            cleaned = safe_text(text, default="")

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

    # Backward compatibility with existing Gemini responses
    # and existing unit tests.
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


# ============================================================
# TECHNICAL PAYLOAD DETECTION
# ============================================================

def product_has_technical_payload(product):
    notes = product.get("retrieval_notes", "")

    if looks_like_technical_payload(notes):
        return True

    name = product.get("product_name", "")

    if looks_like_technical_payload(name):
        return True

    for spec in product.get("key_specifications", []):
        if not isinstance(spec, dict):
            continue

        if looks_like_technical_payload(spec.get("name", "")):
            return True

        if looks_like_technical_payload(spec.get("value", "")):
            return True

        if looks_like_technical_payload(spec.get("evidence", "")):
            return True

    return False


# ============================================================
# NORMALIZE PRODUCT COMPARISON
# ============================================================

def normalize_comparison(comparison, submitted_products):
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

        # Reject technical payloads even if Gemini inferred
        # a product name, specifications, or price.
        if product_has_technical_payload(product):
            normalized_products.append({
                "product_name": "Unavailable",
                "brand": "Unavailable",
                "marketplace": marketplace,
                "url": product_url,
                "retrieval_status": "failed",
                "retrieval_notes": (
                    "The retrieved page contained technical or "
                    "script content instead of usable product information."
                ),
                "variant_match": False,
                "variant_status": "unknown",
                "key_specifications": [],
                "price": {
                    "amount": None,
                    "currency": "INR",
                    "status": "not_found",
                    "evidence": None
                },
                "availability": "Unknown",
                "strengths": [],
                "tradeoffs": []
            })

            continue

        product_name = safe_text(
            product.get("product_name")
        )

        brand = safe_text(
            product.get("brand")
        )

        identity_evidence = product.get("identity_evidence")

        identity_valid = evidence_is_usable(identity_evidence)

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
        # PRICE
        # ----------------------------------------------------

        raw_price = product.get("price", {})

        if not isinstance(raw_price, dict):
            raw_price = {}

        price_amount = raw_price.get("amount")
        price_evidence = raw_price.get("evidence")

        valid_price = price_evidence_is_valid(
            price_amount,
            price_evidence
        )

        if valid_price:
            try:
                price_amount = int(float(price_amount))
            except (TypeError, ValueError):
                price_amount = None
                valid_price = False

        price = {
            "amount": price_amount if valid_price else None,
            "currency": "INR",
            "status": "verified" if valid_price else "not_found",
            "evidence": (
                price_evidence.strip()[:500]
                if valid_price
                else None
            )
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
                "evidence": None
            }

        # ----------------------------------------------------
        # AVAILABILITY
        # ----------------------------------------------------

        availability = safe_text(
            product.get("availability"),
            default="Unknown"
        )

        # ----------------------------------------------------
        # STRENGTHS AND TRADEOFFS
        # ----------------------------------------------------

        # Do not allow unsupported AI-generated claims to
        # appear as factual product advantages/disadvantages.
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
            "key_specifications": specifications,
            "price": price,
            "availability": availability,
            "strengths": strengths,
            "tradeoffs": tradeoffs
        })

    # ========================================================
    # COMPARISON AND RECOMMENDATIONS
    # ========================================================

    verified_products = [
        product
        for product in normalized_products
        if product["retrieval_status"] == "verified"
    ]

    raw_comparison = comparison.get("comparison", {})

    if not isinstance(raw_comparison, dict):
        raw_comparison = {}

    if len(verified_products) < 2:
        feature_comparison = []

        recommendations = {
            "best_overall": "insufficient data",
            "best_value": "insufficient data",
            "best_features": "insufficient data",
            "reason": (
                "At least two products must have sufficient verified "
                "information before recommendations can be generated."
            )
        }

    else:
        feature_comparison = raw_comparison.get(
            "feature_comparison",
            []
        )

        if not isinstance(feature_comparison, list):
            feature_comparison = []

        recommendations = comparison.get("recommendations", {})

        if not isinstance(recommendations, dict):
            recommendations = {}

        allowed_names = {
            product["product_name"]
            for product in verified_products
        }

        def safe_recommendation(value):
            if isinstance(value, str) and value in allowed_names:
                return value

            return "insufficient data"

        recommendations = {
            "best_overall": safe_recommendation(
                recommendations.get("best_overall")
            ),
            "best_value": safe_recommendation(
                recommendations.get("best_value")
            ),
            "best_features": safe_recommendation(
                recommendations.get("best_features")
            )
        }

    # ========================================================
    # PRICE COMPARISON
    # ========================================================

    priced_products = [
        product
        for product in normalized_products
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

    limitations = comparison.get("data_limitations", [])

    if not isinstance(limitations, list):
        limitations = []

    limitations = safe_text_list(limitations)

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

    product_urls = event.get("productUrls")

    submitted_products, error = validate_product_urls(product_urls)

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
            "\nUse Google Search only as a supplementary source. "
            "Do not treat search snippets as proof of a current price "
            "unless the exact price and product variant are clearly "
            "supported by the source."
        )

    return f"""
You are a careful product research and comparison assistant.

Compare the submitted products using the supplied marketplace URLs.

Category: {data["category"]}
Budget in INR: {data["budget"]}
User priority: {data["priority"]}

Submitted products:
{products_json}

Use URL Context to inspect the actual product pages.
{search_instruction}

IMPORTANT EVIDENCE RULES:

1. Never invent product names, brands, specifications, prices,
   availability, or product variants.

2. Every product identity must have identity_evidence copied or
   summarized from the retrieved product page.

3. Every specification must include its own evidence.

4. A specification without page evidence must be omitted.

5. Price must be a numeric INR amount only when the retrieved
   page explicitly supports that amount. Include the price evidence.

6. If price cannot be established, use amount=null,
   status="not_found", evidence=null.

7. Compare the requested model and storage variant.
   A different colour may be treated as a partial match only when
   the model and storage match.

8. Never use a different model or storage variant as an exact match.

9. Do not infer prices from memory, search snippets, old listings,
   product launch prices, or AI knowledge.

10. If a page contains only scripts, JavaScript bundles, monitoring
    payloads, or unrelated technical content, mark retrieval_status
    as "failed" and do not infer product details.

11. A product can be "verified" only when its identity is supported,
    at least two specifications have evidence, and its requested
    model/storage/variant match is established.

12. A different-colour product with matching model and storage can
    be "partial". Explain the colour difference in retrieval_notes.

13. Do not make recommendations when fewer than two products have
    sufficient verified information.

14. Do not put unsupported prices in strengths, tradeoffs,
    comparison text, or recommendations.

Return ONLY valid JSON. No Markdown fences.

Use exactly this structure:

{{
  "products": [
    {{
      "product_name": "Product name",
      "brand": "Brand",
      "identity_evidence": "Evidence for identity",
      "retrieval_status": "verified, partial, or failed",
      "retrieval_notes": "Retrieval details",
      "model_match": true,
      "storage_match": true,
      "color_match": true,
      "variant_match": true,
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

def call_gemini_once(data, api_key, use_search=False):
    prompt = build_gemini_prompt(data, use_search)

    tools = [
        {"url_context": {}}
    ]

    if use_search:
        tools.append({"google_search": {}})

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "Return only valid JSON. Use evidence from "
                    "the supplied product pages. Never invent facts."
                )
            }]
        },
        "contents": [{
            "role": "user",
            "parts": [{
                "text": prompt
            }]
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
        headers={
            "Content-Type": "application/json"
        },
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

    text_parts = [
        part.get("text", "")
        for part in parts
        if isinstance(part, dict)
    ]

    output_text = "\n".join(text_parts).strip()

    if not output_text:
        raise ValueError("Gemini returned an empty response.")

    # Remove accidental Markdown fences.
    output_text = re.sub(
        r"^```(?:json)?\s*",
        "",
        output_text,
        flags=re.IGNORECASE
    )

    output_text = re.sub(
        r"\s*```$",
        "",
        output_text
    )

    return json.loads(output_text), response_data


# ============================================================
# GEMINI FALLBACK
# ============================================================

def needs_search_fallback(comparison):
    products = comparison.get("products", [])

    if not isinstance(products, list):
        return True

    if len(products) < 2:
        return True

    for product in products:
        if not isinstance(product, dict):
            return True

        if not evidence_is_usable(
            product.get("identity_evidence")
        ):
            return True

        if not product.get("key_specifications"):
            return True

        price = product.get("price", {})

        if not isinstance(price, dict):
            return True

        if price.get("amount") is None:
            return True

    return False


def call_gemini(data):
    api_key = get_gemini_api_key()

    first_comparison, first_response = call_gemini_once(
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
        fallback_comparison, fallback_response = call_gemini_once(
            data,
            api_key,
            use_search=True
        )

        # Use the fallback result only if it provides at least
        # as many product records as the original response.
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
            data["products"]
        )

        return response(200, {
            "status": "comparison_generated",
            "model": metadata.get("model", GEMINI_MODEL),
            "dataRetrieved": True,
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
    product_urls = event.get("productUrls")

    submitted, error = validate_product_urls(product_urls)

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

    # Handle API Gateway / Lambda Function URL requests.
    request_context = event.get("requestContext", {})

    if isinstance(request_context, dict):
        http_context = request_context.get("http", {})

        if isinstance(http_context, dict):
            method = http_context.get("method")

            if method == "OPTIONS":
                return response(200, {
                    "status": "success"
                })

            if method and method != "POST":
                return response(405, {
                    "status": "method_not_allowed",
                    "message": "Only POST requests are supported."
                })

    # Parse public Function URL request bodies.
    if "body" in event:
        raw_body = event.get("body")

        if isinstance(raw_body, str):
            if len(raw_body.encode("utf-8")) > MAX_BODY_SIZE:
                return response(413, {
                    "status": "request_too_large",
                    "message": "Request body exceeds the allowed size."
                })

            try:
                event = json.loads(raw_body)
            except json.JSONDecodeError:
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

    # Handle base64-encoded Function URL requests.
    if isinstance(event, dict) and event.get("isBase64Encoded"):
        import base64

        try:
            decoded = base64.b64decode(
                event.get("body", "")
            ).decode("utf-8")

            event = json.loads(decoded)

        except Exception:
            return response(400, {
                "status": "validation_error",
                "message": "Unable to decode request body."
            })

    if not isinstance(event, dict):
        return response(400, {
            "status": "validation_error",
            "message": "Invalid request payload."
        })

    action = event.get("action")

    if action == "ai_summary":
        return handle_ai_summary(event)

    return handle_legacy_validation(event)
