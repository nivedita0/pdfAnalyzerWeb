from typing import Dict, Any, List
import requests
import os
import json

class ZoteroIntegration:
    """Integration with Zotero API for reference management"""
    
    def __init__(self, api_key=None, user_id=None):
        self.api_key = api_key or os.getenv("ZOTERO_API_KEY")
        self.user_id = user_id or os.getenv("ZOTERO_USER_ID")
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
            "collections": [collection_key] if collection_key else []
        }
        
        try:
            response = requests.post(url, headers=headers, json=[zotero_item])
            if response.status_code in (200, 201):
                return True
            else:
                print(f"Error adding item: {response.status_code}")
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