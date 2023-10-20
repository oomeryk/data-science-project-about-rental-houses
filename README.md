# web_scraping--data_analysis--data_visulation--machine_learning
Web scraping on two sites using selenium library in python. Analyzing and visualization the extracted data with pandas - numpy - matplotlib - seaborn libraries. Create machine learning algorithms and showing predicted price and many useful output based on entered, desired rental house informations in Türkiye.

~ENG

To access rental advertisement information from Emlakjet website, list of districts in Türkiye was required. This list was captured from the "https://www.drdatastats.com/turkiye-il-ve-ilceler-listesi/" site using selenium. Then, on the "https://www.emlakjet.com/kiralik-konut/" website, the elements of the district list were searched sequentially with the help of selenium, and all rental housing advertisements in Turkey were extracted and put into an Excel file ([emlakjet.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/12775494/emlakjet.xlsx)). The dataset in the excel file was edited and analyzed using numpy and pandas.   
Edited excel file: [advertss.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/13034418/advertss.xlsx). 
Then, the necessary visualizations were made.

   The slow execution of the selenium code is shown in this video:  https://github.com/oomeryk/web_scraping_and_data_anaysis/assets/127151005/9c0a54c5-61e5-4af8-aade-6d5437ced2dc
   



~TR

Emlakjet sitesinden kiralık ilan bilgilerine ulaşmak için Türkiye'deki ilçeler listesi gerekiyordu. Selenium kullanılarak "https://www.drdatastats.com/turkiye-il-ve-ilceler-listesi/" sitesinden bu liste oluşturuldu. Daha sonra "https://www.emlakjet.com/kiralik-konut/" sitesinde, oluşturulan ilçeler listesinin elemanları selenyum yardımıyla sırayla aratılarak Türkiye'deki tüm kiralık konut ilanları ele geçirilip excel dosyasına ([emlakjet.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/12775494/emlakjet.xlsx)) atıldı. Numpy ve pandas kullanılarak excel dosyasındaki veriseti düzenlenip analiz edildi. 
 
 Düzenlenmiş excel dosyası: [advertss.xlsx](https://github.com/oomeryk/web_scraping_and_data_anaysis/files/13034418/advertss.xlsx). 
 Daha sonra gerekli görselleştirmeler yapıldı.

  Selenium kodunun yavaşlatılmış şekilde çalışması bu videoda gösterilmektedir:  https://github.com/oomeryk/web_scraping_and_data_anaysis/assets/127151005/9c0a54c5-61e5-4af8-aade-6d5437ced2dc

   







