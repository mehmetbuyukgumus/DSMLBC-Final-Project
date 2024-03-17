## Şu anda veri setlerini inceleme ve literatür taraması aşamasındayız.
İşe Yarar Veri Kaynakları

1. Airbnb verileri —> http://insideairbnb.com
2. Fransa’ya dair veri bulmak için —> https://www.data.gouv.fr/fr/
3. Paris’e ait veriler —> https://opendata.paris.fr/pages/home/
4. Londra’yla ilgili veriler —> https://data.london.gov.uk/dataset?q=Transportation

### [Steam Veri Seti](SteamStoreGames)
#### Veri Kaynağı : https://www.kaggle.com/datasets/nikdavis/steam-store-games?select=steamspy_tag_data.csv
Oyunların bilgileri, sistem gereksinimleri, resmi web sayfaları, steam community sayfalarına kadar
detaylı bilgiler içeren bir veri seti. Github dosya limitleri sebebiyle veri setinin tamamını yüklemedim, detayları 
verdiğim kaynak linkinden inceleyebilirsiniz. Çok detaylı incelemedim ancak içerisinde oyularun "fiyat" 
bilgisi yok sanırım. Başka bir kaynaktan fiyatlara erişerek belki bir oyun fiyat tahmini çalışması yapabiliriz
diye düşündüm. Kaagle'da yapılan bir kaç çalışmayı inceledim. Oyun öneri sistemi ve özellik mühendisliği üzerinde
yapılmış çalışmalar var.

### Predict Value of Football Players Using FIFA Video Game Data
#### Makale Linki: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9721908 

Bu makalede yazarlar FIFA'nın video oyununda yer alan oyuncuların değerlerini tahmin etmek için bir makine öğrenkesi
modeliyle çalışmışlar.

Makalede yöntem itibariyle veri ön işleme kısımları geçildikten sonra hedef değişkeni en iyi açıklayan değişkenler
seçilmeye çalışılmış ve sonrasında kurulan modeller bu değişkenler üzerinde yoğunlaştırılmış.

![img.png](../img/img.png)

Ardından dört adet farklı algoritmayla yeniden model kurulmuş ve bunların başarı kriterleri değerlendirilmiş.

![img.png](../img/models_img.png)

### [VGCharts Veri Seti]([VGCarts](vgcharts))
#### Veri Kaynağı: https://www.kaggle.com/datasets/gregorut/videogamesales
Steam veri setiyle birlikte de kullanılabilir içerisinde oyunların isimlerinin yer aldığı bir değişken var. Ayrıca oyunların satış verilerine dair  ve platformalarına dair bilgileri de içeriyor.

### [Steam Oyun Satış Tahmini 2028]([VGCarts](vgcharts))
#### Veri Kaynağı: https://www.statista.com/statistics/547025/steam-game-sales-revenue/
Steam'in 2028'e kadar olan oyun satışı gelirlerinin tahmin verisi. Kısıtlı bir veri seti ancak
geleceğe dönük bir tahmin projesi gerçekleştirirsek referans olabilir diye düşündüm.

### Bank Customer Churn 
#### Veri Kaynağı: https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction
Veri setindeki değişkenler aşina olduğumuz kavramlar (tenure, yaş, cinsiyet, maaş vb) olduğu için üzerinde çalışması kolay olabilir diye düşündüm. Bankacılık sektöründen daha önce bir veri setiyle çalışmadık, portfolyomuzda olsun isteriz belki...

### Wine Quality
#### Veri Kaynağı: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset
Şarapların asiditesi, yoğunluğu, sülfatı vb. değişkenlerine göre sarabın kalitesini 1-10 arası değerlendiren bir veri seti. Literatür taraması yaparken şaraplar ve şarap tadımlarıyla ilgili kültürlenmek keyifli olabilir diye düşünerek bu veri setini seçtim. İçerik olarak da bana oldukça özgün geldi. 

### Salary Prediction Dataset
#### Veri Kaynağı: https://www.kaggle.com/datasets/thedevastator/jobs-dataset-from-glassdoor
2017'de Glassdoor.com sitesinden alınan iş ilanlarından oluşan bir veri seti. Çoğunlukla veri dünyası ile ilgili meslekler barındırıyor. 
İlgili mesleklerin maaş getirilerini tahminlenirken bir yandan da bu mesleklerdeki en önemli değişkenlerin neler olduğunu analiz etmek herkesin ilgisini çekebileceği için sunum aşamasında öne çıkabilir.

### League of Legends Dataset
#### Veri Kaynağı: https://www.kaggle.com/datasets/bobbyscience/league-of-legends-diamond-ranked-games-10-min/data
Yaklaşık 10 bin farklı Elmas ligindeki LoL oyununun ilk 10 dakikası ile ilgili verileri içeren bir veri seti. Oyunu hangi takımın kazancağını tahminleyecek bir model oluşturulabilir.

