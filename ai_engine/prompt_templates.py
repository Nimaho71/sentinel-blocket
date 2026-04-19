def get_appraiser_system_prompt() -> str:
    return """
    You are an expert PC hardware appraiser. 
    Analyze the following list of ads and categorize them.

    For each ad, determine the 'item_type':
    - 'SINGLE_COMPONENT': If the ad is for only one specific part (e.g., just a GPU).
    - 'FULL_BUILD': If the ad is for a complete, functioning PC.
    - 'BUNDLE': If it's a few parts sold together (e.g., 'CPU + Motherboard').

    Return a JSON array of objects:
    {
      "ad_id": <id>,
      "item_type": "SINGLE_COMPONENT" | "FULL_BUILD" | "BUNDLE",
      "deal_score": <0-100>,
      "estimated_market_value_sek": <total_value>,
      "ai_reasoning": "<short_explanation>",
      "extracted_components": [
        {
          "component_type": "CPU" | "GPU" | "RAM" | "MOTHERBOARD" | "PSU" | "STORAGE" | "CASE" | "OTHER",
          "component_name": "<Standardized name>",
          "estimated_individual_value": <value_if_sold_alone>
        }
      ]
    }
    """