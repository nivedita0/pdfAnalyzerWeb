# Academic Paper Search Engine

A web-based search engine that allows users to search for academic papers across multiple sources:
- ArXiv (Computer Science, AI, etc.)
- PubMed (Biomedical and Life Sciences)
- Semantic Scholar Open Research Corpus

## Features

- Clean, modern interface built with TailwindCSS
- Real-time search across multiple academic databases
- Responsive design that works on all devices
- Displays paper titles, authors, abstracts, and direct links
- Loading indicators and error handling

## Setup

1. Clone this repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure PubMed email:
Edit `app.py` and replace `your-email@example.com` with your email address (required for PubMed API access)

5. Run the application:
```bash
python app.py
```

6. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter your search query in the search box
2. Click on one of the source buttons (arXiv, PubMed, or Semantic Scholar)
3. View the results with titles, authors, abstracts, and links to full papers
4. Click "View Paper" to access the original paper

## Technologies Used

- Python 3.x
- Flask
- TailwindCSS
- arxiv package
- biopython
- requests

## Note

This is a basic implementation and may need additional error handling and rate limiting for production use. Some APIs may require authentication or have usage limits. 