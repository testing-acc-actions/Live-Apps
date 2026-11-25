
import json
from urllib.parse import urlparse

ALLOWED_MARKETPLACES = {
    "amazon.in": "Amazon",
    "flipkart.com": "Flipkart"
}

VALID_PRIORITIES = {
    "value_for_money",
    "lowest_price",
    "performance",
    "reliability"
}


def response(status_code, data):
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json"
        },
        "body": json.dumps(data)
    }


def validate_url(url):
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Product URL is empty.")

    parsed = urlparse(url.strip())

    if parsed.scheme != "https":
        raise ValueError("Only HTTPS product URLs are accepted.")

    hostname = (parsed.hostname or "").lower()

    marketplace = None

    for domain, name in ALLOWED_MARKETPLACES.items():
        if hostname == domain or hostname.endswith("." + domain):
            marketplace = name
            break

    if marketplace is None:
        raise ValueError("Only Amazon India and Flipkart URLs are supported.")

    return {
        "url": url.strip(),
        "marketplace": marketplace
    }


def lambda_handler(event, context):

    try:
        # Support both direct Lambda test events
        # and HTTP requests through a Function URL.
        if "body" in event:
            raw_body = event["body"]

            if isinstance(raw_body, str):
                event = json.loads(raw_body)
            elif isinstance(raw_body, dict):
                event = raw_body
            else:
                return response(400, {
                    "error": "Invalid request body."
                })

        product_urls = event.get("productUrls", [])
        budget = event.get("budget")
        priority = event.get("priority")

        if not isinstance(product_urls, list):
            return response(400, {
                "error": "Product URLs must be provided as a list."
            })

        if not 2 <= len(product_urls) <= 5:
            return response(400, {
                "error": "Please provide between 2 and 5 product URLs."
            })

        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or budget <= 0
        ):
            return response(400, {
                "error": "Please provide a valid budget."
            })

        if priority not in VALID_PRIORITIES:
            return response(400, {
                "error": "Invalid recommendation priority."
            })

        valid_products = []
        invalid_products = []

        for index, url in enumerate(product_urls, start=1):
            try:
                product = validate_url(url)
                product["productNumber"] = index
                valid_products.append(product)

            except ValueError as error:
                invalid_products.append({
                    "productNumber": index,
                    "error": str(error)
                })

        if invalid_products:
            return response(400, {
                "status": "correction_required",
                "message": "Please correct the invalid product URLs.",
                "invalidProducts": invalid_products
            })

        return response(200, {
            "status": "validation_success",
            "message": "All product URLs passed initial validation.",
            "budget": budget,
            "priority": priority,
            "totalProducts": len(valid_products),
            "products": valid_products,
            "dataRetrieved": False
        })

    except (json.JSONDecodeError, TypeError, ValueError):
        return response(400, {
            "error": "Invalid request data."
        })

    except Exception:
        print("Unexpected error during request processing.")
        return response(500, {
            "error": "Internal server error."
        })

