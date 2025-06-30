from firecrawl import FirecrawlApp, ScrapeOptions


class FirecrawlService:
    def __init__(self, api_key: str):
        self.app = FirecrawlApp(api_key=api_key)

    def scrape_url(self, url: str, formats: list[str] = ["html", "markdown"]) -> str:
        """
        Scrape the content from the given URL using Firecrawl.

        Args:
            url (str): The URL to scrape.
            options (list[str]): Options for scraping, turned into a ScrapeOptions object.

        Returns:
            str: The scraped content.
        """
        try:
            response = self.app.scrape_url(
                url=url, 
                formats=formats
            )
        except Exception as e:
            raise RuntimeError(f"Error scraping URL {url}: {e}.")
        
        return response   
    
    def crawl_url(self, root_url: str, limit: int = 20, formats: list[str] = ["html", "markdown"]) -> str:
        """
        Crawl the content from the given root URL using Firecrawl.

        Args:
            root_url (str): The root URL to crawl.
            limit (int): The maximum number of pages to crawl.
            formats (list[str]): Options for crawling, turned into a ScrapeOptions object.

        Returns:
            str: The crawled content.
        """
        
        options = ScrapeOptions(formats=formats)

        try:
            response = self.app.crawl_url(
                root_url=root_url, 
                options=options,
                limit=limit
            )
        except Exception as e:
            raise RuntimeError(f"Error crawling URL {root_url}: {e}.")

        return response