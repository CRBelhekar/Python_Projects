## Scrapy Tutorial

We are going to scrape [quotes.toscrape.com](http://quotes.toscrape.com/), a website that lists quotes from famous authors.

This tutorial will walk you through these tasks:

  * Creating a new Scrapy project
  * Writing a spider to crawl a site and extract data
  * Exporting the scraped data using the command line
  * Storing the scraped data
  
### Creating a project

Before you start scraping, you will have to set up a new Scrapy project. Enter a directory where you’d like to store your code and run:
`scrapy startproject tutorial`

This will create a `tutorial` directory with the following contents:
`tutorial/
    scrapy.cfg            # deploy configuration file
    
    tutorial/             # project's Python module, you'll import your code from here
    
        __init__.py
        
        items.py          # project items definition file
        
        middlewares.py    # project middlewares file
        
        pipelines.py      # project pipelines file
        
        settings.py       # project settings file
        
        spiders/          # a directory where you'll later put your spiders
        
            __init__.py`
     
### Creating a spider
Spiders are classes that you define and that Scrapy uses to scrape information from a website (or a group of websites). They must subclass `Spider` and define the initial requests to make, optionally how to follow links in the pages, and how to parse the downloaded page content to extract data.

### Running the spider
`scrapy crawl quotes`

The command runs the spider with name `quotes`.

### Extracting data
`scrapy shell 'http://quotes.toscrape.com/page/1/'`

### Storing the scraped data
`scrapy crawl quotes -o quotes.json`

That will generate an `quotes.json` file containing all scraped items, serialized in JSON.

Learn more about [Scrapy](https://scrapy.org/) in detail [here](https://docs.scrapy.org/en/latest/intro/tutorial.html).
