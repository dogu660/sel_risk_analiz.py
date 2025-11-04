import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import time
import numpy as np
import plotly.express as px 
from math import floor
# Tekrar deneme (Retry) mekanizması için gereken importlar
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 1. Konfigürasyon ve API Bilgileri ---
st.set_page_config(page_title="🌊 Gelişmiş Türkiye Sel Risk Analizi (81 İl)", layout="wide")

# GÜVENLİK VE EN İYİ UYGULAMA: API Anahtarını .streamlit/secrets.toml dosyasından oku
try:
    API_KEY = st.secrets["OPENWEATHER_API_KEY"]
except KeyError:
    # Eğer anahtar bulunamazsa kullanıcıya uyarı gösterilir
    st.error("❌ API Anahtarı bulunamadı! Lütfen `.streamlit/secrets.toml` dosyasını oluşturun ve `OPENWEATHER_API_KEY` değişkenini ekleyin.")
    API_KEY = None 
    
API_URL = "https://api.openweathermap.org/data/2.5/weather"
# İnatçı 10 saniyelik limit sorununu aşmak için zaman aşımı 5 saniyeye düşürüldü.
TIMEOUT_SECS = 5 

# Global bir requests oturumu oluştur (Retry mekanizmasını içerir)
def configure_session():
    """Bağlantı hatalarında otomatik tekrar deneme sağlayan oturum kurar."""
    retry_strategy = Retry(
        total=3,  # Toplam 3 tekrar denemesi (ilk istek + 2 tekrar)
        backoff_factor=1, # Tekrarlar arasında 1, 2, 4 saniye bekler
        status_forcelist=[429, 500, 502, 503, 504], # Sunucu hatalarında tekrar dener
        allowed_methods={"HEAD", "GET", "OPTIONS"}
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    http = requests.Session()
    http.mount("https://", adapter)
    http.mount("http://", adapter)
    return http

# Global oturumu başlat
SESSION = configure_session()

# --- 2. 81 İL LİSTESİ ve KONFİGÜRASYON ---

TUM_ILLER = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", 
    "Aydın", "Balıkesir", "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", 
    "Çanakkale", "Çankırı", "Çorum", "Denizli", "Diyarbakır", "Edirne", "Elazığ", 
    "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkari", 
    "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", 
    "Kırklareli", "Kırşehir", "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", 
    "Kahramanmaraş", "Mardin", "Muğla", "Muş", "Nevşehir", "Niğde", "Ordu", "Rize", 
    "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat", "Trabzon", 
    "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", 
    "Karaman", "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", 
    "Karabük", "Kilis", "Osmaniye", "Düzce"
] 

DEFAULT_RAKIM = 500  
DEFAULT_ALTYAPI = 7.0 

SEHIR_KONFIGURASYON = {}

# Altyapı Verisi Çeşitlendirme (Rastgele varyasyon)
np.random.seed(42) 
for il in TUM_ILLER:
    altyapi_varyasyon = DEFAULT_ALTYAPI + np.random.uniform(-0.5, 0.5) 
    SEHIR_KONFIGURASYON[il] = {"rakim": DEFAULT_RAKIM, "altyapi": round(altyapi_varyasyon, 1)}

# Kritik iller için özel değerler
SEHIR_KONFIGURASYON.update({
    "İstanbul": {"rakim": 100, "altyapi": 6.0},
    "Ankara": {"rakim": 938, "altyapi": 8.5},
    "İzmir": {"rakim": 25, "altyapi": 7.5}, 
    "Antalya": {"rakim": 30, "altyapi": 7.0},
    "Mersin": {"rakim": 15, "altyapi": 6.5},
    "Rize": {"rakim": 10, "altyapi": 5.0}, 
    "Konya": {"rakim": 1021, "altyapi": 9.0}, 
    "Gaziantep": {"rakim": 850, "altyapi": 7.0}, 
})

SEHIRLER = TUM_ILLER

# --- 3. Risk Hesaplama ve Renklendirme Fonksiyonları ---

def yagis_carpani_belirle(yagis_mm):
    """Yağış yoğunluğuna göre ek risk çarpanı belirler."""
    if yagis_mm > 10.0:
        return 1.2
    elif yagis_mm > 5.0:
        return 1.1
    else:
        return 1.0

def sel_riski_hesapla(yagis_mm, bulutluluk, rakım, altyapi):
    """Risk puanını hesaplar (Max 10.0 Puan)."""
    
    # 1. Yağış Temel Puanı (Max 6.0 Puan)
    yagis_capran = yagis_carpani_belirle(yagis_mm)
    temel_yagis = min(yagis_mm / 15.0 * 6.0 * yagis_capran, 6.0)

    # 2. Coğrafi Faktör Puanı (Max 1.5 Puan)
    rakim_f = max((200 - min(rakım, 200)) / 200 * 0.75, 0)
    altyapi_f = (10 - altyapi) / 10.0 * 0.75 
    coğrafi_f = rakim_f + altyapi_f
    
    # 3. Bulutluluk/Belirsizlik Puanı (Max 2.5 Puan)
    if temel_yagis > 0.05:
        bulutluluk_puani = bulutluluk / 100 * 2.5
    else:
        bulutluluk_puani = bulutluluk / 100 * 0.5 

    risk_puani = temel_yagis + coğrafi_f + bulutluluk_puani
    
    return round(min(max(risk_puani, 0.0), 10.0), 2)

def risk_seviyesi_tanimla(risk_puani):
    if risk_puani < 1.5:
        return "ÇOK DÜŞÜK", "KÜÇÜK SU BİRİKİNTİLERİ", "green"
    elif risk_puani < 3.5:
        return "DÜŞÜK/ORTA", "YEREL SU TAŞKINLARI", "lime"
    elif risk_puani < 7.0:
        return "YÜKSEK RİSK", "CİDDİ SEL RİSKİ", "orange"
    else:
        return "ÇOK YÜKSEK", "BÜYÜK ALAN SU BASKINI", "red"

def risk_renk_kodu(risk_seviyesi):
    return {"ÇOK DÜŞÜK": "green", "DÜŞÜK/ORTA": "lime", "YÜKSEK RİSK": "orange", "ÇOK YÜKSEK": "red"}.get(risk_seviyesi, "gray")

# --- 4. Veri Çekme (Retry Mekanizması Kullanılıyor) ---
@st.cache_data(ttl=120) 
def sehir_verisi_getir(sehir, api_key):
    
    if not api_key:
        return None 

    config = SEHIR_KONFIGURASYON.get(sehir)
    rakım = config["rakim"]
    altyapi = config["altyapi"]
    
    params = {"q": sehir + ",TR", "appid": api_key, "units": "metric", "lang": "tr"}

    try:
        # Retry mekanizmalı SESSION.get ve TIMEOUT_SECS = 5 kullanılıyor.
        response = SESSION.get(API_URL, params=params, timeout=TIMEOUT_SECS)
        data = response.json()

        if response.status_code != 200:
            st.error(f"API HATA KODU {response.status_code} - {sehir}: {data.get('message', 'Bilinmeyen Hata')}", icon="⚠️")
            
            return {
                "sehir": sehir, "enlem": None, "boylam": None, "yagis": 0.0,
                "bulutluluk": 0, "rakım": rakım, "altyapi": altyapi,
                "risk_puan": 0.0, "risk": "VERİ YOK", "buyukluk": f"API HATA KODU {response.status_code}", "renk": "gray"
            }
        
        if "coord" not in data:
            st.warning(f"Koordinat verisi alınamadı: {sehir}. Rakım varsayım ({rakım}m) kullanılıyor.", icon="📍")
            enlem, boylam = None, None
        else:
            enlem, boylam = data["coord"]["lat"], data["coord"]["lon"]
        
        yagis = data.get("rain", {}).get("1h", 0.0)
        bulutluluk_yuzdesi = data.get("clouds", {}).get("all", 0)
        
        risk_puani = sel_riski_hesapla(yagis, bulutluluk_yuzdesi, rakım, altyapi)
        risk, buyukluk, renk = risk_seviyesi_tanimla(risk_puani)
        
        return {
            "sehir": sehir, "enlem": enlem, "boylam": boylam, 
            "yagis": yagis, "bulutluluk": bulutluluk_yuzdesi, 
            "rakım": rakım, "altyapi": altyapi, "risk_puan": risk_puani, 
            "risk": risk, "buyukluk": buyukluk, "renk": renk
        }

    except requests.exceptions.RequestException as e:
        # 3 kez tekrar denemeye rağmen hata alınırsa burası çalışır.
        st.error(f"Ağ Hatası/Zaman Aşımı - {sehir}: Bağlantı 3 kez tekrar denendi ve başarısız oldu. Detay: {e}", icon="❌")
        return {
            "sehir": sehir, "enlem": None, "boylam": None, "yagis": 0.0,
            "bulutluluk": 0, "rakım": rakım, "altyapi": altyapi,
            "risk_puan": 0.0, "risk": "VERİ YOK", "buyukluk": "Ağ Hatası/Zaman Aşımı (Retry Başarısız)", "renk": "gray"
        }

# --- 5. Harita Oluşturma ---
def harita_olustur(veriler):
    harita = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="CartoDB positron")

    colormap = {
        "ÇOK DÜŞÜK": 'rgba(0, 128, 0, 0.7)',
        "DÜŞÜK/ORTA": 'rgba(173, 255, 47, 0.8)',
        "YÜKSEK RİSK": 'rgba(255, 140, 0, 0.9)',
        "ÇOK YÜKSEK": 'rgba(255, 0, 0, 1.0)',
        "VERİ YOK": 'rgba(128, 128, 128, 0.5)'
    }

    for veri in veriler:
        if veri["enlem"] is None or veri["risk"] == "VERİ YOK":
            continue

        popup_html = f"""
        <b>{veri['sehir']}</b><br>
        Yağış (1s): {veri['yagis']:.2f} mm/m²<br>
        Bulutluluk: %{veri['bulutluluk']:.0f}<br>
        **Risk Puanı: {veri['risk_puan']:.2f}/10**<br>
        ---<br>
        Sel Riski: <b>{veri['risk']}</b><br>
        Olası Etki: <b>{veri['buyukluk']}</b>
        """
        
        radius = max(5, veri["risk_puan"] * 2)
        fill_color = colormap.get(veri["risk"], 'gray')

        folium.CircleMarker(
            [veri["enlem"], veri["boylam"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            color=fill_color.replace('a', '1').replace(')', ', 1)'),
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.7
        ).add_to(harita)

    return harita

# --- 6. Streamlit Arayüzü ---

if not API_KEY:
    st.title("🌧️ Türkiye Sel Risk Analizi")
    st.header("⚡ Kurulum Hatası")
    st.warning("Lütfen API anahtarınızı `secrets.toml` dosyasına ekleyerek uygulamayı yeniden başlatın.")
else:
    # Veri Çekme
    with st.spinner("🌍 Tüm 81 il için veri ve risk analizi yapılıyor..."):
        tum_veriler = [sehir_verisi_getir(sehir, API_KEY) for sehir in SEHIRLER]
        df_tum = pd.DataFrame([v for v in tum_veriler if v is not None]) # None değerleri filtrele
        
        df_risk = df_tum[df_tum['risk'] != 'VERİ YOK']
        harita_veriler = df_risk[df_risk["enlem"].notna()].to_dict('records')
        api_hatali_sayi = len(SEHIRLER) - len(df_risk)
        
        if not df_risk.empty:
            en_yuksek_risk_puani = df_risk['risk_puan'].max()
            en_riskli_il = df_risk.loc[df_risk['risk_puan'].idxmax(), 'sehir']
            risk_basi = f"| 🔥 {en_riskli_il} ({en_yuksek_risk_puani:.2f} Puan)"
        else:
            risk_basi = ""

    st.title(f"🌧️ Gelişmiş Türkiye Sel Risk Analizi {risk_basi}")

    # Proje Açıklaması
    st.header("⚡ Detaylı Sel Risk Analiz Metodolojisi")
    st.markdown("""
    Bu interaktif panel, Türkiye'deki **81 il** için anlık sel riskini hesaplamak üzere geliştirilmiştir. **Bağlantı Sorunları İçin Otomatik Tekrar Deneme (Retry)** mekanizması ve **düşük zaman aşımı (5s)** ayarı uygulanmıştır.
    """)
    st.divider()

    # --- Sidebar (Yan Panel) Geliştirmesi ---
    st.sidebar.header("🗺️ Harita Filtreleri")
    
    risk_seviyeleri_secenekleri = ["Tümü"] + [k for k in ["ÇOK YÜKSEK", "YÜKSEK RİSK", "DÜŞÜK/ORTA", "ÇOK DÜŞÜK"] if k in df_risk['risk'].unique()]
    
    secilen_risk = st.sidebar.selectbox(
        "Risk Seviyesine Göre Filtrele",
        options=risk_seviyeleri_secenekleri
    )

    if secilen_risk != "Tümü":
        harita_veriler_filtrelenmis = df_risk[df_risk['risk'] == secilen_risk].to_dict('records')
    else:
        harita_veriler_filtrelenmis = harita_veriler

    if st.sidebar.button("🔄 Verileri Şimdi Güncelle"):
        st.toast("Veriler manuel olarak güncelleniyor, önbellek temizleniyor...", icon='⏳')
        st.cache_data.clear() 

    # --- Metrik Kartları ve Harita (Ana Alan) ---
    colA, colB, colC = st.columns(3)
    
    if not df_risk.empty:
        colA.metric(
            "En Yüksek Risk Puanı", 
            f"{en_yuksek_risk_puani:.2f}",
            f"({en_riskli_il})"
        )
        colB.metric(
            "Ortalama Yağış (mm/s)",
            f"{df_risk['yagis'].mean():.2f}",
            f"{df_risk['yagis'].max():.2f} (Max)"
        )
        colC.metric(
            "Veri Alınan İl Sayısı",
            f"{len(df_risk)}",
            f"Toplam 81 İlden"
        )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🗺️ Anlık Risk Haritası")
        if harita_veriler_filtrelenmis:
            st_folium(harita_olustur(harita_veriler_filtrelenmis), width=850, height=550, returned_objects=[])
        else:
            st.warning("Seçilen filtreye uygun veri bulunamadı veya harita verisi yüklenemedi.")


    with col2:
        st.subheader("🚦 Risk Göstergesi")
        st.markdown("""
        - <span style="color:red; font-weight:bold;">ÇOK YÜKSEK (7.0+):</span> Büyük Alan Su Baskını
        - <span style="color:orange; font-weight:bold;">YÜKSEK (3.5-7.0):</span> Ciddi Sel Riski
        - <span style="color:lime; font-weight:bold;">DÜŞÜK/ORTA (1.5-3.5):</span> Yerel Su Taşkınları
        - <span style="color:green; font-weight:bold;">ÇOK DÜŞÜK (0-1.5):</span> Küçük Su Birikintileri
        """, unsafe_allow_html=True)
        
        if api_hatali_sayi > 0:
            st.error(f"⚠️ {api_hatali_sayi} il için kritik API verisi alınamadı.")
        
        st.info("Son güncelleme: " + time.strftime("%H:%M:%S"))

    # --- Risk Dağılım Grafiği (DÜZELTİLDİ) ---
    st.divider()
    
    # 1. Mevcut risk sayımlarını al
    risk_dagilim = df_risk['risk'].value_counts().reset_index()
    risk_dagilim.columns = ['Risk Seviyesi', 'İl Sayısı']

    kategori_sirasi = ["ÇOK DÜŞÜK", "DÜŞÜK/ORTA", "YÜKSEK RİSK", "ÇOK YÜKSEK"]

    # 2. Tüm kategorileri içeren bir DataFrame oluştur
    tum_kategoriler = pd.DataFrame({'Risk Seviyesi': kategori_sirasi})

    # 3. Mevcut sayımlar ile tüm kategorileri birleştir (left join). 
    # Eksik kategorilerde İl Sayısı NaN olacaktır.
    risk_dagilim = pd.merge(tum_kategoriler, risk_dagilim, on='Risk Seviyesi', how='left')

    # 4. İl Sayısı sütunundaki (sayısal) eksik (NaN) değerleri 0 ile doldur.
    # Bu, Kategorik sütunu etkilemediği için TypeError hatası vermez.
    risk_dagilim['İl Sayısı'] = risk_dagilim['İl Sayısı'].fillna(0)

    # 5. Kategorik sıralamayı tekrar ayarla ve sırala
    risk_dagilim['Risk Seviyesi'] = pd.Categorical(risk_dagilim['Risk Seviyesi'], categories=kategori_sirasi, ordered=True)
    risk_dagilim = risk_dagilim.sort_values('Risk Seviyesi')
    
    st.subheader(f"📈 Risk Seviyesi Dağılımı ({len(df_risk)} İl Analiz Edildi)")
    
    fig = px.bar(
        risk_dagilim, 
        x='Risk Seviyesi', 
        y='İl Sayısı', 
        color='Risk Seviyesi', 
        color_discrete_map={k: risk_renk_kodu(k) for k in kategori_sirasi},
        text='İl Sayısı',
        height=350,
        labels={'İl Sayısı': 'İl Sayısı', 'Risk Seviyesi': 'Sel Risk Seviyesi'}
    )
    fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': kategori_sirasi})
    st.plotly_chart(fig, use_container_width=True)

    # --- Şehir Bazlı Risk Puanları ve Özeti ---
    st.divider()
    st.subheader(f"📊 Şehir Bazlı Tüm Risk Verileri ({len(SEHIRLER)} İl)")
    
    df_gosterim = df_tum[[
        "sehir", "yagis", "bulutluluk", "rakım", "altyapi", "risk_puan", "risk", "buyukluk"
    ]].rename(columns={
        "sehir": "Şehir", "yagis": "Yağış (mm/s)", "bulutluluk": "Bulutluluk (%)",
        "rakım": "Rakım (m)", "altyapi": "Altyapı (1-10)", "risk_puan": "Risk Puanı",
        "risk": "Sel Riski", "buyukluk": "Olası Etki",
    })

    def renk_risk_puanı(val):
        if pd.isna(val) or val == 0.0:
            return '' 
        elif val >= 7.0:
            return 'background-color: #ff4c4c; color: white'
        elif val >= 3.5:
            return 'background-color: #ff9900'
        elif val >= 1.5:
            return 'background-color: #ccff33'
        return ''

    # Tabloyu Risk Puanına göre büyükten küçüğe sıralama
    st.dataframe(
        df_gosterim
          .sort_values(by="Risk Puanı", ascending=False)
          .style
          .format({'Yağış (mm/s)': "{:.2f}", 'Risk Puanı': "{:.2f}"}) 
          .applymap(renk_risk_puanı, subset=['Risk Puanı']),
        use_container_width=True,
        hide_index=True 
    )

    st.caption("Veri Alınamayan İller 'Risk Puanı' 0.0 gösterir ve tablonun altındadır. Lütfen Streamlit Logları'nı kontrol edin.")