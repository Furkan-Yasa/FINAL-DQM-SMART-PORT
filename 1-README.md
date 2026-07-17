# BU PROJEDE İZLENEN ADIMLAR ADIM ADIM


# Adım 1: Grid Haritanın ve Temel Ortamın Oluşturulması (5x6 Harita)
Amaç: Derin pekiştirmeli öğrenme için gerekli olan simülasyon ortamının (environment) temelini atmak.

Yapılanlar:

5 satır, 6 sütundan oluşan bir ızgara (grid) harita oluşturuldu.

Liman sahasındaki bloklu alanlar (siyah), vincin giremeyeceği şekilde tanımlandı.

Konteynerlerin yerleştirileceği istif alanı (mavi) belirlendi.

K1'den K4'e kadar toplam 4 adet konteyner, farklı boyutlarda (1x3, 3x1, 2x1, 1x2) ve farklı renk/önceliklerde (Yeşil, Sarı, Kırmızı) haritaya yerleştirildi.

Bu aşamada sadece haritanın doğru göründüğünü ve konteynerlerin birbiriyle çakışmadığını doğrulamak için bir görselleştirme kodu yazıldı.

# Adım 2: Tek Vinç (Vinç-1) için DQN Eğitiminin Başlatılması
Amaç: Tek bir otonom vincin, DQN algoritması ile temel hareket ve taşıma görevlerini öğrenebilirliğini test etmek.

Yapılanlar:

Vinç-1 için 6 temel eylem tanımlandı: Yukarı, Aşağı, Sol, Sağ, Pickup (Al), Drop (Bırak).

Ajanın durumu (state) gözlemleyip, ödül/ceza sinyallerine göre bu eylemlerden birini seçmesi sağlandı.

Rastgele seçilen 2 konteyner "sipariş" olarak belirlendi ve Vinç-1'in bunları alıp istif alanına bırakması istendi.

Temel ödül mekanizması kuruldu: her adımda küçük bir zaman cezası (-0.1), başarılı pickup/drop için ödül, çarpma için ceza.

Eğitim sonucunda Vinç-1'in rastgele hareket etmekten kurtulup, zaman zaman başarılı taşımalar yapmaya başladığı gözlemlendi.

# Adım 3: Operasyonel ve Fiziksel Kuralların Entegrasyonu
Amaç: Sistemi gerçek dünyaya yaklaştırmak için liman operasyonlarındaki karmaşık kısıtları modele dahil etmek.

Yapılanlar (Sırayla):

Öncelik Sıralı Pickup: "Yeşil (Uzun Vade) > Sarı (Orta Vade) > Kırmızı (Kısa Vade)" kuralı getirildi. Ajanın yanlış renkte bir konteyneri almaya çalışması ağır bir şekilde cezalandırıldı (-5).

Rotasyon: 1x3 boyutundaki bir konteynerin istif alanına sığmaması durumunda, ajan tarafından 3x1 olarak döndürülüp yerleştirilebilmesi sağlandı.

Çarpışma (Collision): Vinç-1'in bir yük taşırken diğer konteynerlerin üzerinden geçememesi kuralı eklendi. Bu, ortamı dinamik bir engele sahip hale getirdi.

3 Boyutlu (3B) Üst Üste İstifleme: İstif alanı, sadece bir yüzey olmaktan çıkarılıp, her hücresi bir yığın (stack) tutan 3 boyutlu bir matrise dönüştürüldü. Zemin kat dolduğunda üst katlara yerleştirme yapılabilmesi sağlandı.

İstif Hiyerarşisi: Üst üste koyma işlemine iş kuralı getirildi: Yeşil üstüne Sarı/Kırmızı, Sarı üstüne sadece Kırmızı konulabilir. Bu kurala uyulmaması imkansız hale getirildi.

Görsel İyileştirme: 1. katın üzerine (seviye >= 1) yerleşen tüm konteynerlerin, ayırt edilebilmesi için mor renkte gösterilmesi sağlandı.

# Adım 4: Ölçeklenebilirlik Testi (10x10 Büyük Haritaya Geçiş)
Amaç: Geliştirilen tüm kuralların, daha büyük ve karmaşık bir ortamda çalışabilirliğini ve eğitim süresini analiz etmek.

Yapılanlar:

Harita 10x10 boyutuna büyütüldü. Bloklu alanlar ve 4x4'lük yeni bir istif alanı tanımlandı.

Konteyner sayısı 4'ten 18'e çıkarıldı. Yeşil, sarı ve kırmızı konteynerler çeşitli boyutlarda haritaya yerleştirildi.

Sipariş sayısı 12'ye yükseltildi.

Durum (state) vektörü, artan konteyner ve istif alanı bilgisini kapsayacak şekilde 64 boyutuna genişletildi.

DQN ağ mimarisi büyütüldü (512-512-256-128) ve bellek (replay buffer) kapasitesi artırıldı.

Bulgular: Bu ölçekteki bir problemin eğitim süresinin T4 GPU'da dahi 16 saati aştığı ve ajanın yakınsamakta zorlandığı gözlemlendi. Bu, büyük state-aksiyon uzayının temel bir zorluk olduğunu gösterdi.

# Adım 5: İkinci Vinç (Vinç-2) ile Multi-Agent Sistemine Geçiş

Amaç: Ana operatör Vinç-1'in, önü başka konteynerler tarafından kapatıldığında yaşadığı tıkanma sorununu çözmek için iş birliği yapacak ikinci bir otonom vinç eklemek.

Yapılanlar (İki Aşamada):

Kural Tabanlı (Heuristic) Yardımcı Vinç:

İkinci bir vinç (Vinç-2) ortama eklendi.

Vinç-2, başlangıçta DQN ile değil, el yapımı kurallarla (heuristic) çalışacak şekilde tasarlandı. 

Görevi, Vinç-1'in rotası üzerindeki sipariş dışı konteynerleri (engelleri) tespit edip, onları geçici bir depolama alanına taşımaktı. 

Vinç-1 işini bitirdiğinde, Vinç-2 bu engelleri orijinal yerlerine geri koyuyordu. Bu aşama, ikinci vincin işlevselliğini doğrulamak için bir "kavram kanıtı" niteliğindeydi.

Tam Otonom Multi-Agent DQN:
Kural tabanlı sistemin başarısı kanıtlandıktan sonra, Vinç-2 de kendi DQN ağına sahip bağımsız bir ajana dönüştürüldü.

Artık iki ajan (Vinç-1 ve Vinç-2), aynı ortamdan gözlem yaparak ve ortak bir ödül sinyalini paylaşarak eş zamanlı olarak eğitilmeye başlandı.

Vinç-1 siparişleri toplamayı, Vinç-2 ise yolu açmayı ve iş bitince her şeyi eski haline getirmeyi kendi kendine öğrenmeye başladı.

10×10'luk gridde eğitim süresi katlanarak arttığı için, Vinç‑2'nin engel kaldırma işlemleri ve çift ajanlı DQN eğitimi, 5×10'luk grid üzerinde gerçekleştirilmiştir.


# Son Durum: Nihai Proje
Proje, 10x10'luk bir haritada, 18 konteyner ve 2 otonom DQN ajanı ile tam kapasite çalışabilen bir simülasyon haline geldi.

Sistem; rotasyon, çarpışma, öncelik sıralaması, 3B üst üste istifleme ve hiyerarşi gibi tüm karmaşık kuralları başarıyla yönetmektedir.

Büyük haritadaki eğitim süresi çok uzun olduğu için, sistemin başarısı ve işlerliği daha küçük ölçekli (5x10) bir haritada yapılan eğitimlerle kanıtlanmıştır. 

Bu, projenin raporunda "ölçeklenebilirlik bir zorluktur, ancak sistem mimarisi buna uygundur" tezini güçlü bir şekilde desteklemektedir.

# DENEYSEL SON DURUM SONUÇLARI

10×10’luk büyük harita üzerinde çift DQN ajanı ile yapılan eğitim, durum uzayının büyüklüğü (64 boyut) ve ortamın karmaşıklığı nedeniyle oldukça uzun sürmektedir.
Google Colab ortamında gerçekleştirilen eğitim sırasında, oturum süresi kısıtlamaları ve uzun eğitim süresi (~16+ saat) nedeniyle, eğitim tamamlanamadan oturum sonlanmıştır. 
Bu nedenle, büyük haritaya ait nihai eğitim grafiği ve simülasyon çıktısı elde edilememiştir.
Ancak, aynı mimari ve eğitim prosedürünün daha küçük ölçekli 5×10’luk haritada başarıyla çalıştığı gösterilmiş olup, bu durum sistemin ölçeklenebilir olduğunu fakat büyük ölçekte eğitim için daha güçlü donanım veya daha verimli algoritmalara ihtiyaç duyulduğunu ortaya koymaktadır.
