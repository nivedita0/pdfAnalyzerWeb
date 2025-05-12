from typing import Dict, Any, List
import requests
import os
import json

class ZoteroIntegration:
    """Integration with Zotero API for reference management"""
    
    def __init__(self, api_key=None, user_id=None):
        self.api_key = "NrORzkPOim5uzjbtHP01r55O"
        self.user_id = "17129255"
        self.base_url = "https://api.zotero.org"
        
    def is_configured(self) -> bool:
        """Check if Zotero integration is configured"""
        return bool(self.api_key and self.user_id)
        
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections from Zotero"""
        if not self.is_configured():
            return []
            
        url = f"{self.base_url}/users/{self.user_id}/collections"
        headers = {
            "Zotero-API-Key": self.api_key,
            "Zotero-API-Version": "3"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting collections: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error connecting to Zotero: {e}")
            return []
    
    def add_item(self, metadata: Dict[str, Any], collection_key=None) -> bool:
        """Add a new item to Zotero"""
        if not self.is_configured():
            return False
            
        url = f"{self.base_url}/users/{self.user_id}/items"
        headers = {
            "Zotero-API-Key": self.api_key,
            "Zotero-API-Version": "3",
            "Content-Type": "application/json"
        }
        
        # Format data according to Zotero API requirements
        zotero_item = {
            "itemType": "journalArticle",
            "title": metadata.get("title", "Untitled"),
            "creators": [{"creatorType": "author", "name": author} for author in metadata.get("authors", [])],
            "date": metadata.get("year", ""),
            "DOI": metadata.get("doi", ""),
            "url": metadata.get("url", ""),
            "abstractNote": metadata.get("abstract", ""),
            "publicationTitle": metadata.get("journal", ""),
            "volume": metadata.get("volume", ""),
            "issue": metadata.get("issue", ""),
            "pages": metadata.get("pages", ""),
            "language": metadata.get("language", "en"),
            "collections": [collection_key] if collection_key else []
        }
        
        try:
            # Remove empty fields
            zotero_item = {k: v for k, v in zotero_item.items() if v}
            
            # Make sure creators is not empty
            if not zotero_item.get("creators"):
                zotero_item["creators"] = [{"creatorType": "author", "name": "Unknown"}]
            
            response = requests.post(url, headers=headers, json=[zotero_item])
            print(f"Zotero API Response: {response.status_code} - {response.text}")  # Debug log
            
            if response.status_code in (200, 201):
                return True
            else:
                print(f"Error adding item: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            print(f"Error connecting to Zotero: {e}")
            return False
            
    def upload_attachment(self, item_key: str, file_path: str) -> bool:
        """Upload a file attachment to a Zotero item"""
        if not self.is_configured():
            return False
            
        # Implementation would depend on Zotero API for file uploads
        # This is more complex and would require following their documentation
        return False

    def search_items(self, query: str, item_type: str = None, collection_key: str = None, 
                    sort_by: str = "dateAdded", sort_order: str = "desc", limit: int = 50) -> List[Dict[str, Any]]:
        """Search for items in Zotero library
        
        Args:
            query (str): Search query string
            item_type (str, optional): Filter by item type (e.g., 'journalArticle', 'book', etc.)
            collection_key (str, optional): Filter by collection
            sort_by (str, optional): Field to sort by (default: 'dateAdded')
            sort_order (str, optional): Sort order ('asc' or 'desc', default: 'desc')
            limit (int, optional): Maximum number of results (default: 50)
            
        Returns:
            List[Dict[str, Any]]: List of matching items
        """
        if not self.is_configured():
            return []
            
        url = f"{self.base_url}/users/{self.user_id}/items"
        headers = {
            "Zotero-API-Key": self.api_key,
            "Zotero-API-Version": "3"
        }
        
        # Build query parameters
        params = {
            "q": query,
            "sort": sort_by,
            "direction": sort_order,
            "limit": limit
        }
        
        if item_type:
            params["itemType"] = item_type
            
        if collection_key:
            url = f"{self.base_url}/users/{self.user_id}/collections/{collection_key}/items"
            
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                items = response.json()
                # Process items to extract relevant metadata
                processed_items = []
                for item in items:
                    processed_item = {
                        "key": item.get("key"),
                        "title": item.get("data", {}).get("title", "Untitled"),
                        "authors": [creator.get("name", "") for creator in item.get("data", {}).get("creators", [])],
                        "year": item.get("data", {}).get("date", ""),
                        "type": item.get("data", {}).get("itemType", ""),
                        "abstract": item.get("data", {}).get("abstractNote", ""),
                        "journal": item.get("data", {}).get("publicationTitle", ""),
                        "doi": item.get("data", {}).get("DOI", ""),
                        "url": item.get("data", {}).get("url", "")
                    }
                    processed_items.append(processed_item)
                return processed_items
            else:
                print(f"Error searching items: {response.status_code}")
                return []
        except Exception as e:
            print(f"Error connecting to Zotero: {e}")
            return []

    def get_item_details(self, item_key: str) -> Dict[str, Any]:
        """Get detailed information about a specific item
        
        Args:
            item_key (str): The Zotero item key
            
        Returns:
            Dict[str, Any]: Item details or empty dict if not found
        """
        if not self.is_configured():
            return {}
            
        url = f"{self.base_url}/users/{self.user_id}/items/{item_key}"
        headers = {
            "Zotero-API-Key": self.api_key,
            "Zotero-API-Version": "3"
        }
        
        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                item = response.json()
                return {
                    "key": item.get("key"),
                    "title": item.get("data", {}).get("title", "Untitled"),
                    "authors": [creator.get("name", "") for creator in item.get("data", {}).get("creators", [])],
                    "year": item.get("data", {}).get("date", ""),
                    "type": item.get("data", {}).get("itemType", ""),
                    "abstract": item.get("data", {}).get("abstractNote", ""),
                    "journal": item.get("data", {}).get("publicationTitle", ""),
                    "doi": item.get("data", {}).get("DOI", ""),
                    "url": item.get("data", {}).get("url", ""),
                    "collections": item.get("data", {}).get("collections", []),
                    "tags": [tag.get("tag", "") for tag in item.get("data", {}).get("tags", [])]
                }
            else:
                print(f"Error getting item details: {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error connecting to Zotero: {e}")
            return {}