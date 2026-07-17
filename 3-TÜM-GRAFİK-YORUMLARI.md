# ENGELKALDIRMAMULTIAGENT.png

Bu png'de Çoklu ajan (Multi-Agent) yapısının eğitim sürecine ait ödül grafiği görülmektedir. 

Eğitimin ilk yaklaşık 400 bölümünde ödül değerlerinde belirgin dalgalanmalar meydana gelmiştir. 

Bunun temel nedeni ajanların çevreyi keşfetme aşamasında olması ve farklı hareket stratejilerini denemesidir. 

İlerleyen bölümlerde ödül değerlerinin yaklaşık aynı seviyede kararlı hale geldiği görülmektedir. 

Bu durum iki vincin koordineli çalışmayı öğrendiğini ve başarılı bir politika geliştirdiğini göstermektedir. 

Grafikte zaman zaman görülen ani negatif düşüşler ise başarısız senaryolar veya keşif hareketlerinden kaynaklanmakta olup, genel öğrenme performansını olumsuz etkilememektedir.

# dqn-10x10-graph.png

10×10 liman ortamında DQN algoritmasının eğitim süreci gösterilmektedir.

Eğitimin ilk bölümlerinde ödül değerlerinde dalgalanmalar gözlenirken, yaklaşık 300. bölümden sonra ödüllerin kararlı bir yapıya ulaştığı görülmektedir. 

Bu durum sinir ağının ortam dinamiklerini başarıyla öğrendiğini göstermektedir. 

Eğitim sürecinde görülen kısa süreli negatif sapmalar keşif stratejisinden kaynaklanmakta olup, ajanın kısa sürede tekrar yüksek ödül seviyelerine ulaşması öğrenmenin kararlı şekilde devam ettiğini göstermektedir.

# z_Q-LEARNING.png

Klasik Q-Learning algoritmasının eğitim grafiği görülmektedir. 

Eğitimin başlangıcında toplam ödül değerleri düşük seviyelerde seyretmektedir. 

Ancak ilk yaklaşık 1000 bölüm içerisinde ödül değerlerinin hızlı bir şekilde yükseldiği ve yaklaşık 20.000 seviyesinde kararlı hale geldiği görülmektedir. 

Bu durum ajanın başarılı bir politika öğrendiğini ve hedef konteynerleri etkin şekilde istifleyebildiğini göstermektedir. 

Eğitim boyunca görülen kısa süreli ödül düşüşleri ise keşif mekanizmasının doğal sonucu olup, ajanın genel öğrenme başarısını etkilememektedir.
