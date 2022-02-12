from time import sleep

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

if __name__ == "__main__":
    # browser = webdriver.Chrome(ChromeDriverManager().install())
    browser = webdriver.Chrome("/usr/bin/chromedriver")
