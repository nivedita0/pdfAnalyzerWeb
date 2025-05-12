import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Dict, List, Any, Optional, Tuple
import pdfplumber
from PIL import Image
import io
import base64

class DocumentVisualizer:
    """Combined visualization tool for PDF documents"""
    
    def __init__(self, output_dir: str = "static/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + " "
                return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_and_process_data(self, text: str) -> Tuple[Dict, Dict, Dict]:
        """Extract and process data from text"""
        data = {}
        numeric_data = {}
        non_numeric_data = {}
        
        for line in text.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()

                # Extract numeric values
                numeric_value = re.findall(r"[-+]?\d*\.\d+|\d+", value)
                if numeric_value:
                    numeric_data[key.strip()] = float(numeric_value[0])
                else:
                    non_numeric_data[key.strip()] = value.strip()

        return data, numeric_data, non_numeric_data

    def generate_visualization(self, 
                             data: Dict[str, float], 
                             viz_type: str, 
                             title: str = None) -> Dict[str, Any]:
        """Generate visualization and return as base64 encoded image"""
        try:
            df = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
            
            if df.empty or not df['Value'].notna().any():
                return {"error": "No valid data for visualization"}

            plt.figure(figsize=(12, 6))
            
            if viz_type == "bar":
                sns.barplot(x='Key', y='Value', data=df)
                plt.xticks(rotation=45, ha='right')
                plt.title(title or 'Bar Plot - Data Visualization')
                
            elif viz_type == "pie":
                df = df[df['Value'] > 0]  # Filter out non-positive values
                if not df.empty:
                    plt.pie(df['Value'], labels=df['Key'], autopct='%1.1f%%', startangle=90)
                    plt.title(title or 'Pie Chart - Data Proportions')
                else:
                    return {"error": "No positive values for pie chart"}
                    
            elif viz_type == "line":
                plt.plot(df['Key'], df['Value'], marker='o', linestyle='-', color='b')
                plt.xlabel('Category')
                plt.ylabel('Value')
                plt.title(title or 'Line Plot - Trends Over Categories')
                plt.xticks(rotation=45, ha='right')
                
            elif viz_type == "histogram":
                plt.hist(df['Value'].dropna(), bins=10, color='skyblue', edgecolor='black')
                plt.xlabel('Value')
                plt.ylabel('Frequency')
                plt.title(title or 'Histogram - Distribution of Values')
            
            else:
                return {"error": f"Unknown visualization type: {viz_type}"}

            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return {
                "success": True,
                "image": img_str,
                "data": df.to_dict(orient='records'),
                "type": viz_type
            }
            
        except Exception as e:
            plt.close()
            return {"error": f"Error generating visualization: {str(e)}"}

    def generate_all_visualizations(self, pdf_path: str) -> Dict[str, Any]:
        """Generate all types of visualizations for a document"""
        try:
            # Extract text
            text = self.extract_text_from_pdf(pdf_path)
            if not text:
                return {"error": "Could not extract text from PDF"}

            # Process data
            _, numeric_data, non_numeric_data = self.extract_and_process_data(text)
            
            if not numeric_data:
                return {"error": "No numeric data found for visualization"}

            # Generate all visualization types
            viz_types = ["bar", "pie", "line", "histogram"]
            results = {}
            
            for viz_type in viz_types:
                result = self.generate_visualization(numeric_data, viz_type)
                if "error" not in result:
                    results[viz_type] = result

            return {
                "success": True,
                "visualizations": results,
                "numeric_data": numeric_data,
                "non_numeric_data": non_numeric_data
            }
            
        except Exception as e:
            return {"error": f"Error generating visualizations: {str(e)}"}

    def get_visualization_explanation(self, viz_type: str, data: Dict) -> str:
        """Generate explanation for the visualization"""
        if viz_type == "bar":
            return f"This bar plot shows the comparison of {len(data)} categories. Each bar represents a category's value, making it easy to compare different categories."
        elif viz_type == "pie":
            return f"This pie chart shows the proportion of values across {len(data)} categories. Each slice represents a category's share of the total."
        elif viz_type == "line":
            return f"This line plot shows trends across {len(data)} categories. The line connects points representing each category's value, highlighting patterns and changes."
        elif viz_type == "histogram":
            return f"This histogram shows the distribution of {len(data)} values. Each bar represents a range of values, and its height shows how many values fall within that range."
        return "Visualization explanation not available." 