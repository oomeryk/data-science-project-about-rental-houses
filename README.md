# web_scraping--data_analysis--data_visualization--machine_learning
Data science project consisting of 4 parts:   1-Web Scraping    2-Data Analysis    3-Data Visualization    4-Machine Learning

Web scraping on two sites using selenium library in python. Analyzing and visualization the extracted data with pandas - numpy - matplotlib - seaborn libraries. Creating machine learning algorithms and showing predicted price and many useful output based on entered, desired rental house informations in Türkiye.



~~ To access rental advertisement information from Emlakjet website, list of districts in Türkiye was required. This list was captured from the "https://www.drdatastats.com/turkiye-il-ve-ilceler-listesi/" site using selenium. Then, on the "https://www.emlakjet.com/kiralik-konut/" website, the elements of the district list were searched sequentially with the help of selenium, and all rental housing advertisements in Turkey were extracted and put into an Excel file ([emlakjet.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/12775494/emlakjet.xlsx)). The dataset in the excel file was edited and analyzed using numpy and pandas. Then, the necessary visualization were made using matplotlib and useful outputs were obtained with XGBoost algorithm.


Edited excel file:  [adverts.xlsx](https://github.com/oomeryk/Data-Science-Project/files/13061178/adverts1-2.xlsx). 


The slow execution of the selenium code is shown in this video:  https://github.com/oomeryk/web_scraping_and_data_anaysis/assets/127151005/9c0a54c5-61e5-4af8-aade-6d5437ced2dc

   
Prediction application made by machine learning algorithm:  ![ML-output-screenshot](https://github.com/oomeryk/Data-Science-Project/assets/127151005/cc24700e-0272-4775-8ed0-f527b3270cc2)



~TR

4 bölümden oluşan veri bilimi projesi: 1-Web Kazıma 2-Veri Analizi 3-Veri Görselleştirme 4-Makine Öğrenimi

Python'da Selenium kütüphanesini kullanarak iki sitede web kazıma. Çıkarılan verilerin pandas - numpy - matplotlib - seaborn kütüphaneleriyle analiz edilmesi ve görselleştirilmesi. Türkiye'de girilen, istenen kiralık ev bilgilerine göre makine öğrenimi algoritmaları oluşturmak ve tahmini fiyat ve birçok faydalı çıktıyı göstermek.


Emlakjet sitesinden kiralık ilan bilgilerine ulaşmak için Türkiye'deki ilçeler listesi gerekiyordu. Selenium kullanılarak "https://www.drdatastats.com/turkiye-il-ve-ilceler-listesi/" sitesinden bu liste oluşturuldu. Daha sonra "https://www.emlakjet.com/kiralik-konut/" sitesinde, oluşturulan ilçeler listesinin elemanları selenyum yardımıyla sırayla aratılarak Türkiye'deki tüm kiralık konut ilanları alınıp excel dosyasına ([emlakjet.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/12775494/emlakjet.xlsx)) atıldı. Numpy ve pandas kullanılarak excel dosyasındaki veriseti düzenlenip analiz edildi. Daha sonra matplotlib kütüphanesiyle gerekli görselleştirmeler yapıldı ve XGBoost algoritmasıyla kullanışlı sonuçlar elde edildi.

 
Düzenlenmiş excel dosyası:  [adverts.xlsx](https://github.com/oomeryk/Data-Science-Project/files/13061178/adverts1-2.xlsx). 
 

Selenium kodunun yavaşlatılmış şekilde çalışması bu videoda gösterilmektedir:  https://github.com/oomeryk/web_scraping_and_data_anaysis/assets/127151005/9c0a54c5-61e5-4af8-aade-6d5437ced2dc


Makine öğrenmesi algoritmasıyla yapılan tahmin uygulaması:   ![ML-output-screenshot](https://github.com/oomeryk/Data-Science-Project/assets/127151005/cc24700e-0272-4775-8ed0-f527b3270cc2)







