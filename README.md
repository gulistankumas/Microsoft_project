# Building Your First Local RAG Application with Foundry Local

Kendi belgelerinden kaynak gostererek cevap veren yerel bir terminal uygulamasi.
Microsoft Learn RAG rehberinin belge okuma ve metin parcalama ile genisletilmis
bir staj demosudur. Azure hesabi, API anahtari, web arayuzu veya veritabani gerekmez.

## Ozellikler

- Bir klasordeki TXT, Markdown ve metin iceren PDF dosyalarini okur.
- Uzun metinleri en fazla 800 karakterlik, 120 karakter ortusen parcalara ayirir.
- Foundry Local ile belge ve soru embedding'lerini olusturur.
- Kosinus benzerligiyle en ilgili parcalari bulur; kaynak sayisi ve esigi ayarlanabilir.
- Dosya adi, parca numarasi ve PDF sayfa numarasini kaynak olarak gosterir.
- Cevabi akis halinde yazdirir; arama ve cevap uretme surelerini gosterir.
- Esigi gecen kaynak yoksa sohbet modeline istek yapmaz.
- Belgeleri model indirmeden onizleyebilir.

## Calistirma

Python 3.11+ ve en az 8 GB RAM gerekir. Iki model ayni anda bellekte tutulur;
yeterli **bos RAM** de gerekir. Ilk calistirmada model ve calistirma bilesenleri
indirilir. Internet, diskte bos alan ve indirme icin zaman gerekir.
Modeller daha sonra yerel onbellekten kullanilir; inference cihazda yapilir.

[uv](https://docs.astral.sh/uv/getting-started/installation/) kurulu olmali.
Asagidaki komutlar proje icin ayri Python ortamini olusturur ve uygulamayi baslatir.
Proje klasorunde PowerShell ile:

```powershell
uv sync --locked
uv run main.py --inspect
uv run main.py
```

`--inspect` sadece belgeleri kontrol eder; Foundry Local'i baslatmaz ve model
indirmez. Python paketleri eksikse `uv` bunlari yine kurabilir.
Windows'ta WinML SDK kullanilir; ayri Foundry CLI kurulumu gerekmez.
Varsayilan olarak CPU modelleri kullanilir. Bu bilgisayarda otomatik secilen
CUDA modeli cevap uretirken GPU kaynak ayirma hatasi verdigi icin CPU tercih edildi.
Uygun GPU/NPU ve surucusu olan cihazlarda `--device auto` ile SDK'nin otomatik
donanim secimini kullanabilirsin. CPU ve GPU varyantlari ayri indirilir.

## Kendi Belgelerin

Belgelerini `documents` klasorune ekleyip uygulamayi baslat. TXT ve Markdown
dosyalari UTF-8 kodlamali olmali. Alt klasorler taranmaz. Calisirken belgeyi
degistirirsen indeksi yenilemek icin uygulamayi yeniden baslat.

Baska bir klasor ve farkli arama ayarlari da kullanabilirsin:

```powershell
uv run main.py --docs "C:\Belgelerim" --top-k 2 --min-score 0.4
```

| Ayar | Varsayilan | Aciklama |
| --- | --- | --- |
| `--docs` | Projenin `documents` klasoru | Bilgi tabaninin konumu |
| `--top-k` | `3` | En fazla kaynak sayisi; 1-5 arasinda |
| `--min-score` | `0.35` | Kosinus benzerligi esigi; -1 ile 1 arasinda |
| `--device` | `cpu` | CPU kullan veya `auto` ile SDK'nin cihaz secimini kullan |
| `--inspect` | Kapali | Ilk bes parcayi modelleri baslatmadan gosterir |

**Benzerlik skoru dogruluk olasiligi degildir.** Varsayilan esik bir demo
ayaridir; veriler uzerinde kalibre edilmemistir. Esigi artirmak daha fazla
sonucu eler ama ilgili belgeleri de eleyebilir. Esigi gecen bir kaynak,
sorunun cevabini mutlaka icermez.

Kisisel veya gizli belgelerini Git'e eklememeye dikkat et.

## Demo Senaryosu

Ornek bilgi tabani, Foundry Local notlari ve hayali Atlas projesini icerir.
**Atlas belgesindeki takvim ve kisiler tamamen ornektir; Microsoft'un resmi
staj kurallari degildir.**

```text
What are the Atlas project deliverables?
When is the Atlas review meeting?
Does RAG retrain the language model?
What is the Atlas project budget?
quit
```

Ilk sorularda kaynak metnini ve modelin cevabini karsilastir. Butce sorusunun
cevabi belgelerde yoktur: modelden bunu belirtmesi istenir, ancak bu davranis
garanti edilmez. Kaynaklar benzerlik esigini gecmezse uygulama cevap uretmez.

Turkce soru sorabilirsin. Kucuk model ve Ingilizce ornek belgeler nedeniyle
ilk demo icin Ingilizce sorular daha uygundur. Her soru bagimsizdir;
sohbet gecmisi tutulmaz. Cikmak icin `quit`, `exit` veya Ctrl+C kullan.

## Mimari

```text
TXT / MD / PDF -> Metin cikarma -> Ortusen parcalar -> Embedding -> Bellekte indeks
Soru -> Embedding -> Kosinus aramasi -> Esik + ilk K kaynak -> Yerel model -> Yanit
```

- Embedding modeli: `qwen3-embedding-0.6b`
- Cevap modeli: `qwen2.5-0.5b`
- `documents.py`: belge okuma, parcalama ve kaynak bilgileri.
- `main.py`: terminal, embedding, arama, prompt ve cevap uretimi.
- `tests/test_rag.py`: modelsiz otomatik testler.

Indeks her baslatmada bellekte yeniden olusturulur. Belgeler bir bulut
sohbet servisine gonderilmez; Foundry Local SDK'nin yerel istemcileri kullanilir.

## Testler

```powershell
uv run python -m unittest discover -s tests -v
```

Testler dosya okuma, PDF hata durumlari, parcalama, kaynak bilgileri, siralama,
esik davranisi ve terminal akislarini kontrol eder. Model istemcileri taklit
edilir; model indirilmez. Bunlar gercek modelin cevap kalitesini olcmez.

## Sinirlar

- Dosya basina 10 MB, PDF basina 100 sayfa, bilgi tabani icin toplam 300 parca siniri var.
- Bozuk, sifreli veya metinsiz dosyalar uyariyla atlanir. Taranmis PDF icin OCR yoktur.
- Karakter tabanli parcalama kullanilir; PDF tablo/kolon okuma sirasi kusurlu olabilir.
- Kaynak numaralari ve cevap dogrulugu otomatik dogrulanmaz. Kaynaklari kontrol et.
- Prompt talimatlari, belgelerdeki kotu niyetli yonlendirmelere karsi kesin koruma saglamaz.
- Buyuk koleksiyonlar ve cok kullanicili kullanim icin tasarlanmamistir.
- Bellek hatasinda diger uygulamalari kapat. Indirme/katalog hatasinda interneti
  ve resmi rehberdeki model adlarini kontrol et.

## Resmi Kaynak

Microsoft Learn: [Build a RAG application](https://learn.microsoft.com/en-us/azure/foundry-local/tutorials/tutorial-build-rag-app).
