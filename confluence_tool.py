import json
import httpx
from typing import Optional
import os
from dotenv import load_dotenv


# Simple function to search Confluence
def search_confluence_docs(query: str, space_key: Optional[str] = None) -> str:
    """
    Search Confluence for documents related to the query.

    Args:
        query: The search query for finding relevant Confluence pages
        space_key: Optional space key to limit search to a specific space

    Returns:
        str: JSON string with search results containing titles, URLs, and excerpts
    """
    # Load environment variables
    load_dotenv()

    # Get Confluence credentials
    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_username = os.getenv("CONFLUENCE_USERNAME")
    confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")
    default_space_key = os.getenv("CONFLUENCE_SPACE_KEY")

    if not all([confluence_url, confluence_username, confluence_api_token]):
        return json.dumps({
            "error": "Missing Confluence configuration",
            "status": "error",
            "message": "Confluence URL, username, or API token not provided"
        })

    try:
        # Construct CQL query
        cql = f'text ~ "{query}"'
        if space_key or default_space_key:
            space = space_key or default_space_key
            cql += f' AND space = "{space}"'

        # Make the search request directly
        response = httpx.get(
            f"{confluence_url.rstrip('/')}/rest/api/content/search",
            params={
                "cql": cql,
                "limit": 10,
                "expand": "metadata.labels,body.view.value"
            },
            auth=(confluence_username, confluence_api_token),
            timeout=30.0
        )

        # Check response status
        response.raise_for_status()
        results = response.json()

        # Format results for readability
        formatted_results = []
        for result in results.get("results", []):
            # Extract content excerpt
            body_content = result.get("body", {}).get("view", {}).get("value", "")
            # Basic HTML to text conversion for excerpt
            excerpt = _extract_excerpt(body_content)

            formatted_results.append({
                "title": result.get("title", "Untitled"),
                "url": f"{confluence_url.rstrip('/')}{result.get('_links', {}).get('webui', '')}",
                "excerpt": excerpt,
                "id": result.get("id"),
                "type": result.get("type")
            })

        return json.dumps({
            "total": results.get("size", 0),
            "results": formatted_results
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error",
            "message": f"Failed to search Confluence: {str(e)}"
        })


# Separate function to retrieve page content
def retrieve_confluence_page(page_id: str) -> str:
    """
    Get the full content of a specific Confluence page.

    Args:
        page_id: The ID of the Confluence page

    Returns:
        str: JSON string with page title, URL, and full content
    """
    # Load environment variables
    load_dotenv()

    # Get Confluence credentials
    confluence_url = os.getenv("CONFLUENCE_URL")
    confluence_username = os.getenv("CONFLUENCE_USERNAME")
    confluence_api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not all([confluence_url, confluence_username, confluence_api_token]):
        return json.dumps({
            "error": "Missing Confluence configuration",
            "status": "error",
            "message": "Confluence URL, username, or API token not provided"
        })

    try:
        response = httpx.get(
            f"{confluence_url.rstrip('/')}/rest/api/content/{page_id}",
            params={"expand": "body.storage,metadata.labels"},
            auth=(confluence_username, confluence_api_token),
            timeout=30.0
        )

        response.raise_for_status()
        page = response.json()

        content = page.get("body", {}).get("storage", {}).get("value", "")

        return json.dumps({
            "title": page.get("title", "Untitled"),
            "url": f"{confluence_url.rstrip('/')}{page.get('_links', {}).get('webui', '')}",
            "content": content,
            "id": page.get("id"),
            "type": page.get("type")
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error",
            "message": f"Failed to retrieve page content: {str(e)}"
        })


# Helper function for extracting text from HTML
def _extract_excerpt(html_content: str, max_length: int = 200) -> str:
    """Extract a plain text excerpt from HTML content"""
    import re
    text = re.sub(r'<[^>]+>', ' ', html_content)
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) > max_length:
        return text[:max_length] + "..."
    return text