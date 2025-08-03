# LungSight: Yapay Zeka Destekli Göğüs Hastalıkları Teşhis Destek Sistemi

Bu proje, tıp profesyonelleri için geliştirilmiş yapay zeka destekli bir göğüs hastalıkları teşhis destek sistemidir. Temel amacı, klinik tanı ve tedavi süreçlerini hızlandırarak sağlık hizmetlerinin verimliliğini artırmaktır. Sistem, özellikle akciğer hastalıklarının erken teşhisine odaklanmakta ve bu doğrultuda çeşitli analiz modülleri sunmaktadır.

LungSight’ın merkezinde, farklı tıbbi verileri analiz edebilen gelişmiş yapay zeka modelleri bulunmaktadır. Bu modeller arasında, oskültasyon (dinleme) yoluyla elde edilen solunum seslerini değerlendirerek astım, KOAH, bronşit ve pnömoni gibi hastalıkların ayırıcı tanısına yardımcı olan bir modül yer alır. Bir diğer önemli modül ise X-ray görüntülerinin analizidir; bu modül, konjenital anomaliler, KOAH, mediastinal kitleler, plevral efüzyon, akciğer iltihabı, pnömotoraks, tüberküloz ve akciğer tümörleri dahil olmak üzere dokuz farklı akciğer patolojisini tespit etme kapasitesine sahiptir. Ayrıca, özellikle akciğer kanserinin erken teşhisine yönelik, X-ray görüntüleri üzerinden benign (iyi huylu) ve malign (kötü huylu) oluşumları ayırt edebilen özel bir onkolojik değerlendirme modülü de mevcuttur. Sistem, bu analizler sonucunda tahmin edilen hastalıkları ve modelin güven skorlarını kullanıcıya sunar.

## Oskültasyon (Dinleme) Sesleri ile Hastalık Tespiti

![Image](https://github.com/user-attachments/assets/3e33d3bd-c618-49cd-bf06-ff5e88740182)

Bu sayfa, kullanıcıların yükledikleri solunum sesi kayıtlarını analiz ederek belirli solunum yolu hastalıklarının (Sağlıklı, Astım, Bronşit, KOAH, Akciğer İltihabı) ön tespitini yapmayı amaçlar. Yapay zeka destekli bir model kullanarak erken teşhise yardımcı olmayı ve kullanıcıları potansiyel sağlık sorunları hakkında bilgilendirmeyi hedefler.

Sayfa Yapısı ve Özellikleri:

1. Kısa Açıklama ve Bilgilendirme:
    - Sayfanın en başında, modülün ne işe yaradığı ve hangi hastalıkları tespit edebileceği hakkında kısa bir bilgilendirme metni bulunur.
    - Yapay zeka destekli analizin erken teşhis ve tedavi şansını artırabileceği vurgulanır.
2. Örnek Dosya veya Demo:
    - Kullanıcıların sistemi deneyebilmesi için örnek solunum sesi dosyalarını (ornek_wav.zip) indirebilecekleri bir bölüm mevcuttur.
3. Tahmin Formu:
    - Kullanıcıların solunum sesi dosyalarını (audio/* formatında) yükleyebilecekleri bir form içerir.
    - “Tahmin Yap” butonu ile analiz süreci başlatılır.
4. Ses Dalga Formu Görselleştirmesi (Dinamik):
    - Kullanıcı bir ses dosyası yüklediğinde, bu dosyanın ses dalga formu bir grafik üzerinde görselleştirilir.
    - Bu özellik, Chart.js kütüphanesi kullanılarak JavaScript ile dinamik olarak oluşturulur.
    - Ses verisi işlenir, gerekirse örnekleme oranı düşürülerek (downsample) grafiğe uygun hale getirilir.
    - Grafik, sesin genlik (amplitude) değişimini zamanla gösterir.
5. Tahmin Sonucu Açıklama Alanı (Dinamik):
    - Tahmin yapıldıktan sonra, tespit edilen hastalık (veya sağlıklı durumu) bu alanda gösterilir.
    - Tespit edilen her bir hastalık (Astım, Bronşit, KOAH, Akciğer İltihabı, Sağlıklı) için kısa bir açıklama ve bilgilendirme metni sunulur. Bu metinler, hastalığın ne olduğu ve erken teşhisin önemi gibi konulara değinir.
6. Yükleme ve Sonuç Geçmişi:
    - Kullanıcının daha önce yaptığı tahminlerin bir listesi bu bölümde gösterilir.
    - Her bir geçmiş kaydı için tahmin tarihi, yüklenen dosya adı ve tahmin sonucu listelenir.
    - Eğer geçmiş kayıt yoksa, “Henüz tahmin geçmişi bulunmamaktadır.” mesajı gösterilir.
7. Tahmin İstatistikleri:
    - Yapılan tahminlerin istatistiksel bir özeti grafiksel olarak sunulur.

## X-ray Görüntüleri ile Hastalık Tespiti

![Image](https://github.com/user-attachments/assets/faa54470-5a21-48df-8a9b-e4c56c3d5b90)

Bu sayfa, kullanıcıların yükledikleri X-ray (röntgen) görüntülerini analiz ederek çeşitli akciğer ve göğüs hastalıklarının (Sağlıklı, Konjenital Anomali, KOAH, Mediastinal Kitle, Plevral Efüzyon, Akciğer İltihabı, Akciğer Sönmesi (Pneumotorax), Tüberküloz, Akciğer Tümörü) ön tespitini yapmayı amaçlar. Yapay zeka destekli bir model kullanarak erken teşhise yardımcı olmayı ve kullanıcıları potansiyel sağlık sorunları hakkında bilgilendirmeyi hedefler.

Sayfa Yapısı ve Özellikleri:

1. Kısa Açıklama ve Bilgilendirme:
    - Sayfanın başında, modülün ne işe yaradığı ve hangi hastalıkları tespit edebileceği hakkında bir bilgilendirme metni bulunur.
    - Tespit edilebilecek hastalıkların bir listesi sunulur.
    - Yapay zeka destekli analizin erken teşhis ve tedavi şansını artırabileceği vurgulanır.
2. Örnek Dosya veya Demo:
    - Kullanıcıların sistemi deneyebilmesi için örnek X-ray görüntülerini (ornek_xray.zip) indirebilecekleri bir bölüm mevcuttur.
3. Tahmin Formu:
    - Kullanıcıların X-ray görüntü dosyalarını (image/* formatında, genellikle JPG veya PNG) yükleyebilecekleri bir form içerir.
    - “Tahmin Yap” butonu ile analiz süreci başlatılır.
4. X-ray Görüntü Önizleme (Dinamik):
    - Kullanıcı bir görüntü dosyası yüklediğinde, bu görüntü sayfada önizlenir.
    - Bu özellik JavaScript ile dinamik olarak çalışır. Yüklenen dosya FileReader ile okunur ve bir <img> elementinin src özelliğine atanarak görüntülenir.
    - Görüntünün adı ve boyutu gibi bilgiler de (imageInfo) gösterilebilir.
    - Önizleme alanı (previewContainer) başlangıçta gizlidir ve dosya seçildiğinde görünür hale gelir.
5. Tahmin Sonucu Açıklama Alanı (Dinamik):
    - Tahmin yapıldıktan sonra, tespit edilen hastalık (veya sağlıklı durumu) bu alanda gösterilir.
    - Tespit edilen her bir hastalık için (yukarıda listelenenler) kısa bir açıklama ve bilgilendirme metni sunulur. Bu metinler, hastalığın ne olduğu, potansiyel riskleri ve uzman görüşünün önemi gibi konulara değinir.
6. Yükleme ve Sonuç Geçmişi:
    - Kullanıcının daha önce yaptığı X-ray tahminlerinin bir listesi bu bölümde gösterilir.
    - Her bir geçmiş kaydı için tahmin tarihi, yüklenen dosya adı ve tahmin sonucu listelenir.
    - Eğer geçmiş kayıt yoksa, “Henüz tahmin geçmişi bulunmamaktadır.” mesajı gösterilir.
7. Tahmin İstatistikleri:
    - Yapılan X-ray tahminlerinin istatistiksel bir özeti grafiksel olarak sunulur.

## X-ray Görüntüleri ile Onkolojik Hastalık Tespiti

![Image](https://github.com/user-attachments/assets/42b446a7-67a9-4ae9-b729-2ecc14435524)

Bu sayfa, kullanıcıların yükledikleri X-ray (röntgen) görüntülerini analiz ederek akciğer kanseri riskini değerlendirmeyi amaçlar. Yapay zeka modeli, yüklenen görüntüye dayanarak üç olası sonuçtan birini tahmin eder: Sağlıklı, Benign (İyi Huylu Tümör) veya Malignant (Kötü Huylu Tümör). Sayfanın temel hedefi, erken teşhise katkıda bulunmak ve kullanıcıları potansiyel riskler konusunda bilgilendirerek uzman bir doktora başvurmalarını teşvik etmektir.

Sayfa Yapısı ve Özellikleri:

1. Kısa Açıklama ve Bilgilendirme:
    - Sayfanın başında, modülün X-ray görüntüleri üzerinden akciğer kanseri riskini nasıl değerlendirdiği açıklanır.
    - Tespit edilebilecek üç ana kategori (Sağlıklı, Benign, Malignant) listelenir.
    - Yapay zeka destekli analizin erken teşhis ve tedavi şansını artırabileceği vurgulanır.
2. Örnek Dosya veya Demo:
    - Kullanıcıların sistemi ve analiz sürecini deneyebilmeleri için örnek onkolojik X-ray görüntülerini (ornek_onkolojik_xray.zip) indirebilecekleri bir bağlantı sunulur.
3. Tahmin Formu:
    - Kullanıcıların X-ray görüntü dosyalarını (image/* formatında) yükleyebilecekleri bir form bulunur.
    - “Tahmin Yap” butonu ile yüklenen görüntü analiz için sunucuya gönderilir.
4. X-ray Görüntü Önizleme (Dinamik):
    - Kullanıcı bir görüntü dosyası seçtiğinde, bu görüntü sayfada önizlenir.
    - Bu özellik JavaScript ile çalışır: FileReader API’si kullanılarak dosya okunur ve bir <img> elementinin src özelliğine atanarak görüntülenir.
    - Önizleme alanı (previewContainer) başlangıçta gizlidir ve geçerli bir görüntü dosyası seçildiğinde görünür hale gelir.
    - JavaScript tarafında dosya türü kontrolü (file.type.startsWith('image/')) yapılarak sadece görüntü dosyalarının yüklenmesi sağlanır.
5. Tahmin Sonucu Açıklama Alanı (Dinamik)
6. Yükleme ve Sonuç Geçmişi:
    - Kullanıcının daha önce yaptığı akciğer kanseri tahminlerinin bir listesi bu bölümde tablo formatında gösterilir.
    - Her bir geçmiş kaydı için tahmin tarihi, yüklenen dosya adı ve tahmin sonucu listelenir.
    - Eğer geçmiş kayıt yoksa, “Henüz tahmin geçmişi bulunmamaktadır.” mesajı gösterilir.
7. Tahmin İstatistikleri

## Sağlık Asistanı

![Image](https://github.com/user-attachments/assets/dd64a350-33ea-4370-8e87-1b0bf607ae0c)

Bu sayfa, kullanıcılara Türkçe dilinde yapay zeka destekli bir sağlık asistanı sunar. “Gemini Türkçe Sağlık Asistanı” olarak adlandırılan bu sohbet botu, Google’ın Gemini dil modelini kullanarak akciğer sağlığı ve genel tıbbi konular hakkında soruları yanıtlamak üzere tasarlanmıştır. Sayfanın temel amacı, kullanıcılara bilgilendirici yanıtlar sağlamaktır; ancak verilen bilgilerin tıbbi tavsiye yerine geçmediği ve her zaman bir doktora danışılması gerektiği açıkça belirtilir.

## Hasta Takip Modülü

![Image](https://github.com/user-attachments/assets/d5a0d94b-a484-43e5-8be3-1a2ce6033aa2)

Kapsamlı bir klinik araç olarak tasarlanan LungSight, hasta yönetimi özelliklerini de içerir. Tıp profesyonelleri, sisteme yeni hastalar kaydedebilir, mevcut hastaların listesini görüntüleyebilir ve her bir hastanın detaylı kişisel bilgilerine, tıbbi kayıtlarına ve teşhis geçmişine erişebilir. Yeni tıbbi kayıtların ve teşhislerin sisteme girilmesi de mümkündür.

## Rapor ve Analitik

Raporlama ve istatistiksel analiz, LungSight’ın önemli bir diğer bileşenidir. Kullanıcılar, belirli tarih aralıkları ve veri türlerine (hasta kayıtları, tahmin sonuçları, teşhisler) göre özelleştirilmiş raporlar oluşturabilir ve bu raporları görüntüleyebilir. Sistem, genel bir bakış sunan bir gösterge paneli aracılığıyla toplam hasta sayısı, yapılan tahminler ve konulan teşhisler gibi önemli metrikleri sergiler. Ayrıca, tahminlerin doğruluk oranları, hastalık dağılımları ve hasta demografisi gibi konularda detaylı istatistiksel analizler ve grafiksel sunumlar da sağlar.
