from requests import get
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json
import re
import requests
import threading
from preprocess import Preprocessor
	
def synchronized(func):
	
    func.__lock__ = threading.Lock()
		
    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func

class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': '*/*',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'
    url_rg = r'https:\/\/www\.imdb\.com\/title\/(\w*)\/?'
    content_reg = r'__NEXT_DATA__" type="application\/json">(.*"scriptLoader":\[\]})'
    title_url = 'https://www.imdb.com/title/'
    link_reg = r'<a href="(\/review\/\w+\/\?ref_=tt_urv)"\n>Permalink<\/a>'
    map_reg = r'<script type="application\/ld\+json">({.*\n})<\/script>'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """

        self.crawling_threshold = crawling_threshold
        self.frontier = []
        self.read_from_file_as_json()
        self.dynamic_ids = self.crawled.copy()

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        with open('IMDB_crawled_ids.json', 'w+') as f:
            f.write(json.dumps(self.crawled))
            f.close()

        with open('IMDB_not_crawled.json', 'w+') as f:
            f.write(json.dumps(self.not_crawled))
            f.close()

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        with open('IMDB_crawled_ids.json', 'r') as f:
            c = f.readline()
            self.crawled = [] if c == '' else json.loads(c)

        with open('IMDB_not_crawled.json', 'r') as f:
            c = f.readline()
            self.not_crawled = [] if c == '' else json.loads(c)


    def crawl(self, url):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The response of the get request
        """
        t = requests.get(url, headers=self.headers)
        t.raise_for_status()
        content = t.text

        j =  {}
        try:
            response = re.search(self.content_reg, content).group(1)
            j = json.loads(response)
        except:
            pass
        links = re.findall(self.link_reg, content)
        reviews = []
        for l in links:
            t1 = requests.get('https://www.imdb.com' + l)
            t1.raise_for_status()
            link_content = t1.text
            r = json.loads(re.search(self.map_reg, link_content, re.U | re.DOTALL).group(1))
            if 'reviewBody' not in r.keys():
                ss = '<div class="text show-more__control">'
                s = link_content.find(ss)
                e = link_content.find('<div class="actions text-muted">')
                r['reviewBody'] = link_content[s + len(ss):e].replace('</div>', '')
            reviews.append(r)
        j['reviews'] = reviews
        return j

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        result = self.crawl(self.top_250_URL)['props']['pageProps']['pageData']['chartTitles']['edges']
        res_str = []
        for i in range(250):
            id =  result[i]['node']['id']
            if id not in self.crawled:
                if id not in self.not_crawled:
                    self.not_crawled.append(id)
        self.write_to_file_as_json()



    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        WHILE_LOOP_CONSTRAINTS = None
        NEW_URL = None
        THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        futures = []
        crawled_counter = 0
        with ThreadPoolExecutor(20) as executor:
            while self.crawling_threshold > len(self.dynamic_ids):
                if (len(self.not_crawled) == 0):
                    wait(futures)
                self.frontier = self.safe_subsequence()
                for id in self.frontier:
                    self.dynamic_ids.append(id)
                    futures.append(executor.submit(self.crawl_page_info, id))
            wait(futures)
            for e in self.dynamic_ids:
                if e not in self.crawled:
                    self.crawl_page_info(id)
        self.write_data()

    batch_size = 20

    def safe_subsequence(self):
        res = []
        for _ in range(self.batch_size):
            try:
                res.append(self.not_crawled.pop(0))
            except:
                return res
        return res
    
    def get_url(self, id, suffix=''):
        return self.title_url + id + suffix + '/'

    def crawl_page_info(self, id):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        movie_url, review_url, summ_url = self.get_url(id), self.get_url(id, '/reviews'), self.get_url(id, '/plotsummary')
        print("new iteration. crawling id:", id)
        while True:
            try:
                movie = self.get_imdb_instance()
                res = self.extract_movie_info(self.crawl(movie_url), movie)
                if res == -1:
                    print('restricted movie', id)
                    return
                self.extract_review_info(self.crawl(review_url), movie)
                self.extract_summary_info(self.crawl(summ_url), movie)
                break
            except:
                print('ERROR occurred during crawling. retrying... ', id)
        # with open(f'data/data_{id}.json', 'w') as f:
        #     f.write(json.dumps(movie, indent=1))
        self.finalize_doc(movie)
        print(f'doc {id} added to list')
        return
    
    data = []

    @synchronized
    def finalize_doc(self, movie):
        self.crawled.append(movie['id'])
        for id in movie['related_links']:
            if self.needs_crawl(id):
                print('new id found!', id)
                self.not_crawled.append(id)
        self.data.append(movie)
        if len(self.data) >= 20:
            self.write_data()
            
        self.write_to_file_as_json()

    def write_data(self):
        print('writing to file...')
        with open('IMDB_crawled.json', 'r') as f:
                j = json.load(f)
                j.extend(self.data)
                f.close()
        with open('IMDB_crawled.json', 'w') as f:
            json.dump(j, f, indent=1, ensure_ascii=False)
            f.close()
        self.data.clear()
    dynamic_ids = []
    def needs_crawl(self, id):
        return id not in self.dynamic_ids and id not in self.not_crawled


    def extract_movie_info(self, res, movie):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: dict
            The JSON response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        data = res['props']['pageProps']['aboveTheFoldData']
        if data['productionStatus'] and data['productionStatus']['restriction'] and data['productionStatus']['restriction']['restrictionReason']:
            return -1
        main_data = res['props']['pageProps']['mainColumnData']
        movie['id'] = res['props']['pageProps']['tconst']
        movie['title'] = self.beautify(data['titleText']['text']) if data['titleText'] else ''
        movie['Image_URL'] = data['primaryImage']['url'] if data['primaryImage'] else None
        movie['first_page_summary'] = self.beautify(data['plot']['plotText']['plainText']) if data['plot'] and data['plot']['plotText'] else ''
        movie['release_year'] = str(data['releaseYear']['year']) if data['releaseYear'] else ''
        movie['mpaa'] = data['certificate']['rating'] if data['certificate'] else ""
        movie['budget'] = str(main_data['productionBudget']['budget']['amount']) if main_data['productionBudget'] else ""
        movie['gross_worldwide'] = str(main_data['worldwideGross']['total']['amount']) if main_data['worldwideGross'] else ''
        movie['directors'] = [i['name']['nameText']['text'] for i in main_data['directors'][0]['credits']] if main_data['directors'] and len(main_data['directors']) > 0 else []
        movie['writers'] = [i['name']['nameText']['text'] for i in main_data['writers'][0]['credits']] if main_data['writers'] and len(main_data['writers']) > 0 else []
        movie['stars'] = [i['node']['name']['nameText']['text'] for i in main_data['cast']['edges']] if main_data['cast'] else []
        movie['related_links'] = [i['node']['id']  for i in main_data['moreLikeThisTitles']['edges'] if i['node']['canHaveEpisodes'] == False] if main_data['moreLikeThisTitles'] else []
        movie['genres'] = [i['text'] for i in data['genres']['genres']] if data['genres'] else []
        movie['languages'] = [i['text'] for i in main_data['spokenLanguages']['spokenLanguages']] if main_data['spokenLanguages'] else []
        movie['countries_of_origin'] = [i['text'] for i in main_data['countriesOfOrigin']['countries']] if main_data['countriesOfOrigin'] else []
        movie['rating'] = str(data['ratingsSummary']['aggregateRating']) if data['ratingsSummary'] else ''
        return 0

    def beautify(self, s):
        text = re.sub(r'<a[^>]*>(.*?)<\/a>', r'\1', s)
        return text.replace('&#39;', "'").replace('&#39;', "'").replace('&amp;', '&').replace('&gt;', '>').replace('&lt;', '<').replace('<br/><br/>', '').replace('&quot;', '').replace('\n', '').replace('\\n', '').replace('<ul>', '').replace('</ul>', '').replace('<li>', '').replace('</li>', '').replace('<u>', '').replace('</u>', '')

    def extract_review_info(self, response, movie):
        movie['reviews'] = [[self.beautify(i['reviewBody'])] for i in response['reviews']]

    def extract_summary_info(self, response, movie):
        data = response['props']['pageProps']['contentData']['categories']
        for e in data:
            if e['id'] == 'summaries':
                movie['summaries'] = [self.beautify(re.sub(r'<span style=.*$', '', i['htmlContent'])) for i in e['section']['items']]

            elif e['id'] == 'synopsis':
                movie['synopsis'] = [self.beautify(e['section']['items'][0]['htmlContent'])] if len(e['section']['items']) > 0 else ['']


def main():
    imdb_crawler = IMDbCrawler(crawling_threshold=1040)
    imdb_crawler.start_crawling()
    documents = None
    with open('IMDB_crawled.json', 'r') as f:
            documents = json.load(f)
            f.close()
    print('preprocessing...')
    preprocessor = Preprocessor(documents)
    with open('IMDB_crawled_pre_processed.json', 'w+') as f:
        documents = json.dump(preprocessor.preprocess(), f, indent=1)
        f.close()

if __name__ == '__main__':
    main()