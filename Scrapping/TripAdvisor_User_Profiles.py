#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
# import requests
import csv
import re
import time
from selenium import webdriver

# import the webdriver, chrome driver is recommended
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')


# import pymongo
# from pymongo import MongoClient


# In[2]:


# client = MongoClient("localhost", 27017)
# db = client['tripadvisor2']
# profiles = db.reviews100


# In[3]:


# function to check if the button is on the page, to avoid miss-click problem
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True


# time.sleep(2)


def allreviews(URL):
    global review, elementCount
    reviewsList = []
    i = 1
    while driver.find_elements_by_xpath("//div[@style='position:relative']/div"):
        i += 1
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        if i == 20:
            break;

    print('end')
    #     driver.find_element_by_xpath('//*[@id="_evidon-accept-button"]').click()
    try:
        driver.find_element_by_xpath('//button[@class="_3L3LNeQW _39EfpzKn UXe6zT9I _1C95-Ec1"]').click()
        onepage = False
        time.sleep(3)
    except NoSuchElementException:
        onepage = True
        print('no more button')

    #     lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    #     match=False
    #     while(match==False):
    #         lastCount = lenOfPage
    #         time.sleep(2)
    #         lenOfPage = driver.execute_script("window.scrollTo(0, document.body.scrollHeight);var lenOfPage=document.body.scrollHeight;return lenOfPage;")
    #         if lastCount==lenOfPage:
    #             match=True

    review = []
    if onepage == False:
        i = 1
        while driver.find_elements_by_xpath("//div[@style='position:relative']/div") and len(review) < 20:
            #         print(i)
            # time.sleep(2)
            print(len(review))
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            review = driver.find_elements_by_xpath("//div[@style='position:relative']/div")
            time.sleep(3)
        # i+=1

    review = driver.find_elements_by_xpath("//div[@style='position:relative']/div")
    elementCount = len(review)

    #   print(elementCount)
    #     if (elementCount is endcount):
    #         print('end')
    #         break
    #     else:
    #         endcount = elementCount
    #         continue

    print('count elements: ', elementCount)
    for j in range(elementCount):
        try:

            ratingDiv = review[j].find_element_by_xpath(".//div[contains(@class, '_1VhUEi8g _2K4zZcBv')]")
            source_code = ratingDiv.get_attribute("outerHTML")

            reviewScore = int(
                ratingDiv.find_element_by_xpath(".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
                    "class").split("_")[3])

            ratingGeneral = review[j].find_element_by_xpath(".//div[contains(@class, 'ui_poi_review_rating')]")
            ratingGeneralScore = int(ratingGeneral.find_element_by_xpath(
                ".//span[contains(@class, 'ui_bubble_rating bubble_')]").get_attribute(
                "class").split("_")[3])

            #             print(ratingGeneralScore)

            linkReviewLocation = review[j].find_element_by_xpath(
                ".//div[contains(@class, '_2X5tM2jP _2RdXRsdL _1gafur1D')]")
            linkReviewLocation = linkReviewLocation.find_element_by_xpath(".//a[contains(@class, '')]").get_attribute(
                'href')
            #             print(linkReviewLocation)

            #             name = review[j].find_element_by_xpath(".//div[contains(@class, '_2fxQ4TOx')]").text
            reviewTitle = review[j].find_element_by_xpath(".//div[contains(@class, '_3IEJ3tAK _2K4zZcBv')]").text
            reviewDate = review[j].find_element_by_xpath(".//div[contains(@class, '_3Coh9OJA')]").text
            reviewDate = reviewDate.split(': ')[1]
            reviewFor = review[j].find_element_by_xpath(".//div[contains(@class, '_2ys8zX0p ui_link')]").text
            reviewsummary = review[j].find_element_by_xpath(".//div[contains(@class, '_1kKLd-3D')]/a").get_attribute(
                "href")
            reviewLocation = review[j].find_element_by_xpath(".//div[contains(@class, '_20BneOSW')]").text
            reviewCategory = review[j].find_element_by_class_name("_25zALESU").get_attribute("class").split()[1]

            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(reviewsummary)

            time.sleep(2)

            if (check_exists_by_xpath("//span[@class='fullText hidden']")):
                readMore = driver.find_elements_by_xpath("//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                readMore[2].click()
                reviewText = readMore[1].text

            elif (check_exists_by_xpath("//span[@class='fullText ']")):

                readMore = driver.find_elements_by_xpath("//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                reviewText = readMore[0].text

            else:
                reviewdetails = driver.find_elements_by_xpath(
                    "//div[@class='reviewSelector']/div/div[2]/div/div/div[3]/div/p")
                reviewText = reviewdetails[0].text

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            reviewTotal = {'title': reviewTitle,
                           'text': reviewText,
                           'date': reviewDate, 'place': reviewFor,
                           'placeURL': linkReviewLocation,
                           'placeLocation': reviewLocation,
                           'placeType': reviewCategory,
                           'reviewScore': reviewScore / 10.0,
                           'avgScore': ratingGeneralScore / 10.0}
            #             print(reviewTotal)
            reviewTotal = list(reviewTotal.values())
            #             print(reviewTotal)
            reviewsList.append(reviewTotal)
            print('review retieved')


        except:
            print('review not found')

    return reviewsList


def getallReviewsBymainUrl(URL):
    # open user profile
    driver.get(URL)

    time.sleep(10)

    #     driver.maximize_window()
    # accept cookies
    driver.find_element_by_xpath('//*[@id="_evidon-accept-button"]').click()

    # get review count
    count = driver.find_element_by_class_name("_1q4H5LOk").text.replace(",", "")

    # get location for main page
    #     try:
    #         location = driver.find_element_by_class_name("_2VknwlEe._3J15flPT.default").text
    #     except NoSuchElementException:
    #         location = 'NaN'

    #     endcount = int(maxcount) if int(count) > int(maxcount) else int(count)

    reviewsList = allreviews(URL)

    print('count reviews: ' + str(len(reviewsList)))

    return reviewsList


# In[4]:


def create_new_userfile(fname):
    with open(fname, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'text', 'date', 'place', 'placeURL',
                         'placeLocation', 'placeType', 'reviewScore',
                         'avgScore'])


def write_user_reviews(fname, reviewsList):
    with open(fname, 'a', newline='') as result_file:
        wr = csv.writer(result_file)
        for review in reviewsList:
            try:
                wr.writerow(review)
            except:
                print("An exception occurred")


# In[5]:


import pandas as pd

# driver = webdriver.Chrome(r"C:\Users\andre\Downloads\chromedriver_win32\chromedriver.exe")
# maxcount = 10000
i = 0

usersDF = pd.read_csv('users_with_demo.csv')
path = 'https://www.tripadvisor.com/Profile/'

for i in range(len(usersDF)):
    #     print('user: ', i)
    username = usersDF.iloc[i]['username']
    check = usersDF.iloc[i]['check']
    #     print(username)
    if check == False and i >= 4000:
        print('user: ', i)
        print(username)
        filename = str(username) + '.csv'
        create_new_userfile(filename)
        #        driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", chr$
        # driver = webdriver.Chrome("/usr/lib/chromium-browser/chromedriver", chrome_options=chrome_options)

        driver = webdriver.Chrome(r"C:\Users\Andreas\Desktop\chromedriver.exe")
        url = path + str(username) + '?tab=reviews'
        #         location, reviews = getallReviewsBymainUrl(url)
        reviews = getallReviewsBymainUrl(url)

        #         user_profile = {'username': username, 'location': location, 'reviews': reviews}
        #         print(user_profile)
        #         x = profiles.insert_one(user_profile)

        if reviews is not None:
            write_user_reviews(filename, reviews)
            print('OK ' + str(username) + ' reviews inserted!')
            usersDF.at[i, 'check'] = True

        else:
            print('Empty List of Reviews')
        usersDF.to_csv('users_with_demo.csv', index=True)
        driver.close()

    # In[ ]:
