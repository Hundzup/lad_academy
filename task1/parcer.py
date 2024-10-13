import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup

class Parse_hh:
    def __init__(self, url):
        self.url = url
        self.header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        self.page = "&page="
        self.count_v = 0
        
    def parse_url(self):
        data = []
        sum_vacancy = 0        
        i = 0
        while True: 
            response = requests.get(self.url+self.page+str(i), headers=self.header)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = soup.find_all(class_ = 'magritte-card___bhGKz_6-1-5 magritte-card-border-radius-24___o72BE_6-1-5 magritte-card-stretched___0Uc0J_6-1-5 magritte-card-action___4A43B_6-1-5 magritte-card-shadow-on-hover___BoRL3_6-1-5 magritte-card-with-border___3KsrG_6-1-5')
                for article in articles:
                    title = article.find(class_ = "magritte-text___tkzIl_4-3-2").text
                    salary = article.find(class_ = "magritte-text___pbpft_3-0-15 magritte-text_style-primary___AQ7MW_3-0-15 magritte-text_typography-label-1-regular___pi3R-_3-0-15")
                    if salary != None:
                        salary = salary.text
                    city = article.find(attrs={"data-qa": "vacancy-serp__vacancy-address"}).text
                    company = article.find(class_ = 'magritte-text___pbpft_3-0-15 magritte-text_style-primary___AQ7MW_3-0-15 magritte-text_typography-label-3-regular___Nhtlp_3-0-15').text
                    data.append({"title" : title,
                        "salary": salary,
                        "city" : city,
                        'company': company}
                                )
                sum_vacancy += len(articles)
                # print(sum_vacancy)
            else:
                print(f'Ошибка {response.status_code},\n ответ {response.text},\n история {response.history}')
            if len(articles) == 0:
                break
            i += 1
        print(f'Найденно {sum_vacancy} вакансий')
        data = pd.DataFrame(data=data)
        self.count_v = sum_vacancy
        return data