import os 
from dotenv import load_dotenv
from loguru import logger
import openai
from time import time,sleep
import numpy as np
import json
import cohere 
import re

load_dotenv()



class NLPClient:
    def __init__(self):
        self.co = self.get_nlp_client()

    
    def get_nlp_client(self):
        return cohere.Client('COHERE_API_KEY')
    
    
    def open_file(self,filepath):
        with open(filepath,'r',encoding='utf-8') as infile:
            return infile.read()
    
    def cohere_completion(prompt, engine='command-xlarge-20221108',, temp=0.0, top_k=0, top_p=1.0, tokens=400, freq_pen=0.0, pres_pen=0.0, stop=['USER:']):
        max_retry = 5
        retry = 0
        prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
        while True:
            try:
                response = self.co.generate(
                    model=engine,
                    prompt=prompt,
                    max_tokens=tokens,
                    temperature=temp,
                    k=top_k,
                    p=top_p,
                    frequency_penalty=freq_pen,
                    presence_penalty=pres_pen,
                    stop_sequences=stop,
                    return_likelihoods='NONE')
                text = response.generations[0].text.strip()
                text = re.sub('\s+',' ',text)
                return text 

            except Exception as oops:
                retry += 1
                if retry >= max_retry:
                    return 'Cohere error: %s' % oops
                print('Error communicating with Cohere:',oops)
                exit()
                sleep(1)

    
    def get_completion(self,block):
        prompt = self.open_file('../prompt_senti.txt').replace('<<CONVERSATION>>',block)
        logger.debug(f'The final prompt! \n {prompt}')
        response = self.cohere_completion(prompt)
        logger.debug(f'The final response from bot: {response}')
        return response
