"""Validate and clean extracted receipt data."""

import re
from typing import Any, Dict, List, Optional


def clean_price(price_str: Any) -> Optional[float]:
    """
    Clean and convert price string to float.
    
    Args:
        price_str: Price as string, number, or None
    
    Returns:
        Cleaned price as float or None
    """
    if price_str is None:
        return None
    
    if isinstance(price_str, (int, float)):
        return float(price_str)
    
    if not isinstance(price_str, str):
        return None
    
    # Remove currency symbols and whitespace
    cleaned = re.sub(r'[^\d.,-]', '', str(price_str).strip())
    
    # Handle different decimal separators
    cleaned = cleaned.replace(',', '.')
    
    # Remove multiple dots (keep last one)
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
    
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def clean_text(text: Any) -> Optional[str]:
    """
    Clean text string.
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text or None
    """
    if text is None:
        return None
    
    if not isinstance(text, str):
        text = str(text)
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Remove special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s\.,\-\(\)]', '', cleaned)
    
    return cleaned.strip() if cleaned.strip() else None


def validate_receipt_data(receipt_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and clean receipt data.
    
    Args:
        receipt_data: Raw receipt data from OCR
    
    Returns:
        Validated and cleaned receipt data
    """
    validated = {
        "shop_name": clean_text(receipt_data.get("shop_name")),
        "payment_mode": clean_text(receipt_data.get("payment_mode")),
        "total_amount": clean_price(receipt_data.get("total_amount")),
        "items": [],
        "receipt_pathfile": receipt_data.get("receipt_pathfile"),
    }
    
    # Validate and clean items
    items = receipt_data.get("items", [])
    if not isinstance(items, list):
        items = []
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        cleaned_item = {
            "name": clean_text(item.get("name")),
            "price": clean_price(item.get("price")),
        }
        
        # Only add items with valid names
        if cleaned_item["name"]:
            validated["items"].append(cleaned_item)
    
    # Validate total amount matches sum of items (with tolerance)
    if validated["total_amount"] and validated["items"]:
        items_sum = sum(item.get("price", 0) or 0 for item in validated["items"])
        difference = abs(validated["total_amount"] - items_sum)
        
        # If difference is significant, log warning but don't fail
        if difference > 0.01:  # More than 1 cent difference
            print(f"⚠️ Warning: Total amount ({validated['total_amount']}) doesn't match sum of items ({items_sum:.2f})")
    
    return validated


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON from text that may contain markdown or other formatting.
    
    Args:
        text: Text that may contain JSON
    
    Returns:
        Extracted JSON string or None
    """
    if not text:
        return None
    
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object
    start = text.find('{')
    if start == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    end = start
    
    for i, char in enumerate(text[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    if brace_count != 0:
        return None
    
    return text[start:end]


