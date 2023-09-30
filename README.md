# Is-Bankas-ML-Challenge-5
---
Team Members: Hakan Taştan, Furkan Ümit Akkaya


Türkiye İş Bankası ML Challenge #5
Task:
Bir mobil uygulamanın kullanım verileriyle bir yapay zeka modeli geliştirerek her bir kullanıcıya o uygulamada en çok ihtiyaç duyacağı menüyü önermenizdir. Veri setinde hedef olarak, her bir kullanıcının geçmiş kullanımında tercih ettiği menüleri binary olarak belirten 9 tane menü yer almaktadır. Bu 9 menüden hangi 3 tanesinin kullanıcı arayüzüne ekleneceğini tahminlemeniz gerekmektedir.

Evaluation
Metrik kriteri olarak Jaccard Score belirlenmiştir. Her bir menünün önerilip önerilmeyeceğini gösteren 9 adet binary değerden oluşan tahminleriniz bu metrik ile kıyaslanacaktır.

Örnek çıktı şu şekildedir:
Müşteri A -> 000101010 (Müşteri A’ya 4., 6., ve 8. Menüler önerilmesi tahminlenmiştir.)
Müşteri B -> 100000011 (Müşteri A’ya 1., 8., ve 9. Menüler önerilmesi tahminlenmiştir.)
