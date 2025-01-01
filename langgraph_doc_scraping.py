import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time

def normalize_url(url):
    """Normalize URLs to treat similar URLs as the same."""
    parsed = urlparse(url)
    # Remove trailing slashes, fragments (#), and query parameters (?)
    path = parsed.path.rstrip('/')
    return f"{parsed.scheme}://{parsed.netloc}{path}"

def is_valid_domain(url, base_domain):
    """Check if URL belongs to the specified domain."""
    parsed_url = urlparse(url)
    return parsed_url.netloc == base_domain

def scrape_langchain_docs(start_url):
    """Scrape the LangChain documentation and save to text file."""
    base_domain = "langchain-ai.github.io"
    visited_urls = set()  # For tracking scraped URLs
    urls_to_visit = [normalize_url(start_url)]  # Start with normalized URL
    
    # For debugging - keep track of where we found each URL
    url_sources = {}
    
    with open("langchain_docs.txt", "w", encoding="utf-8") as f:
        f.write("LangChain Documentation\n")
        f.write("=" * 50 + "\n\n")
        
        while urls_to_visit:
            current_url = urls_to_visit.pop(0)
            normalized_current_url = normalize_url(current_url)
            
            # Debug print to show URL processing
            print(f"\nProcessing: {current_url}")
            print(f"Normalized: {normalized_current_url}")
            if normalized_current_url in visited_urls:
                print(f"Already visited - skipping")
                continue
            
            # Skip if not in our domain
            if not is_valid_domain(normalized_current_url, base_domain):
                print(f"Outside domain - skipping")
                continue
                
            try:
                # Add delay to be respectful
                time.sleep(1)
                
                response = requests.get(current_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Write the current page content to file
                f.write(f"URL: {current_url}\n")
                f.write(f"Title: {soup.title.string if soup.title else 'No title'}\n")
                f.write("Content:\n")
                f.write(soup.get_text(separator='\n', strip=True))
                f.write("\n" + "=" * 50 + "\n\n")
                
                # Find all links on the page
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(current_url, href)
                        normalized_url = normalize_url(absolute_url)
                        
                        # Debug - store where we found this URL
                        url_sources[normalized_url] = current_url
                        
                        # Only add URLs that are in our domain and haven't been processed
                        if (is_valid_domain(normalized_url, base_domain) and 
                            normalized_url not in visited_urls and 
                            normalized_url not in [normalize_url(u) for u in urls_to_visit]):
                            print(f"Adding new URL: {absolute_url}")
                            urls_to_visit.append(absolute_url)
                        else:
                            print(f"Skipping URL: {absolute_url}")
                            if normalized_url in visited_urls:
                                print(f"  Reason: Already visited")
                            elif not is_valid_domain(normalized_url, base_domain):
                                print(f"  Reason: Outside domain")
                            else:
                                print(f"  Reason: Already in queue")
                
                # Mark current URL as visited using normalized form
                visited_urls.add(normalized_current_url)
                
            except Exception as e:
                print(f"Error scraping {current_url}: {str(e)}")
                continue
        
        # Write summary at the end
        f.write("\nScraping Summary:\n")
        f.write(f"Total pages scraped: {len(visited_urls)}\n")
        
        # Write URL sources for verification
        f.write("\nURL Sources (for verification):\n")
        for url, source in url_sources.items():
            f.write(f"URL: {url}\n")
            f.write(f"Found on: {source}\n")
            f.write("-" * 30 + "\n")

if __name__ == "__main__":
    start_url = "https://langchain-ai.github.io/langgraph/"
    scrape_langchain_docs(start_url)