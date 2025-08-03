from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.http import quote_header_value
import os
from dotenv import load_dotenv
import numpy as np
import librosa
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import google.generativeai as genai
import datetime
import pandas as pd
# TKinter hatasını önlemek için matplotlib backend'ini ayarlayalım
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmeyen backend kullanımı
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sqlalchemy import func, extract, and_
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
# PDF oluşturma için gerekli importlar
import tempfile
import os
import unicodedata

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'mysql+mysqlconnector://root:@localhost/lungsight')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Gemini API anahtarınızı buraya ekleyin
genai.configure(api_key=api_key)

# Gemini modelini yapılandırın
model = genai.GenerativeModel('gemini-2.0-flash')

# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Prediction model
class Prediction(db.Model):
    __tablename__ = 'respiratory_prediction'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    
    user = db.relationship('User', backref=db.backref('predictions', lazy=True))

# Lung Disease Prediction model
class LungDiseasePrediction(db.Model):
    __tablename__ = 'lung_diseases_prediction'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    
    user = db.relationship('User', backref=db.backref('lung_disease_predictions', lazy=True))

# Lung Cancer Prediction model
class LungCancerPrediction(db.Model):
    __tablename__ = 'lung_cancer_prediction'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    prediction_date = db.Column(db.DateTime, nullable=False, default=db.func.current_timestamp())
    
    user = db.relationship('User', backref=db.backref('lung_cancer_predictions', lazy=True))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('prediction'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False

        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            flash('Lütfen e-posta ve şifrenizi kontrol edin.', 'error')
            return redirect(url_for('login'))

        login_user(user, remember=remember)
        return redirect(url_for('prediction'))

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password_confirm = request.form.get('password-confirm')

        if password != password_confirm:
            flash('Şifreler eşleşmiyor.', 'error')
            return redirect(url_for('register'))

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Bu e-posta adresi zaten kayıtlı.', 'error')
            return redirect(url_for('register'))

        new_user = User(name=name, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash('Kayıt başarılı! Şimdi giriş yapabilirsiniz.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/prediction')
@login_required
def prediction():
    return render_template('prediction.html')

def extract_features(file_path, sr=22050, n_mfcc=40, max_len=259, target_duration=6.0):
    y, _ = librosa.load(file_path, sr=sr)
    target_length = int(sr * target_duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    y = (y - np.mean(y)) / (np.std(y) + 1e-9)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).T
    mel = librosa.feature.melspectrogram(y=y, sr=sr).T
    spectral_con = librosa.feature.spectral_contrast(y=y, sr=sr).T
    def fix_shape(feat, max_len):
        if feat.shape[0] < max_len:
            pad_width = max_len - feat.shape[0]
            return np.pad(feat, ((0, pad_width), (0, 0)), mode='constant')
        return feat[:max_len]
    mfcc = fix_shape(mfcc, max_len)
    chroma = fix_shape(chroma, max_len)
    mel = fix_shape(mel, max_len)
    spectral_con = fix_shape(spectral_con, max_len)
    features = np.concatenate([mfcc, chroma, mel, spectral_con], axis=1)
    features = np.array(features)
    features = features[..., np.newaxis]
    return features

audio_model = tf.keras.models.load_model(r'C:\Users\berka\OneDrive\Masaüstü\lungsight\models\audio_81_acc.keras', 
                                 custom_objects={'InputLayer': tf.keras.layers.InputLayer})
audio_numbers_map = {0: 'Astım', 1: 'Bronşit', 2: 'KOAH', 3: 'Sağlıklı', 4: 'Akciğer İltihabı'}

lung_diseases_model = tf.keras.models.load_model(r'C:\Users\berka\OneDrive\Masaüstü\lungsight\models\lung_disease_9_classes.keras')

lung_diseases_numbers_map = {0: 'Sağlıklı', 1: 'Konjenital Anomali', 2: 'KOAH', 3: 'Mediastinal Kitle', 
              4: 'Plevral Efüzyon', 5: 'Akciğer İltihabı', 6: 'Akciğer Sönmesi (Pneumotorax)', 
              7: 'Tüberküloz', 8: 'Akciğer Tümörü'}

lung_cancer_model = tf.keras.models.load_model(r'C:\Users\berka\OneDrive\Masaüstü\lungsight\models\lung_cancer_98_acc.keras')

lung_cancer_numbers_map = {0: 'Sağlıklı', 1: 'Benign (İyi Huylu)', 2: 'Malignant (Kötü Huylu)'}


@app.route('/respiratory_prediction', methods=['GET', 'POST'])
@login_required
def respiratory_prediction():
    prediction_result = None
    waveform_data = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)
            
            # Extract waveform data for visualization
            y, sr = librosa.load(file_path)
            # Downsample the waveform to reduce data size
            waveform_data = y[::100].tolist()
            
            features = extract_features(file_path)
            prediction = audio_model.predict(features[np.newaxis, ...])
            predicted_label = audio_numbers_map[int(np.argmax(prediction))]
            
            # Save prediction to database
            new_prediction = Prediction(
                user_id=current_user.id,
                filename=filename,
                prediction_result=predicted_label
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            os.remove(file_path)
            prediction_result = predicted_label
            
    # Get user's last 5 predictions
    prediction_history = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.prediction_date.desc()).limit(5).all()
    
    # Get prediction statistics
    prediction_stats = db.session.query(
        Prediction.prediction_result,
        db.func.count(Prediction.id)
    ).filter_by(user_id=current_user.id).group_by(Prediction.prediction_result).all()
    
    # Convert stats to dictionary format for the chart
    stats_dict = {label: 0 for label in audio_numbers_map.values()}
    for result, count in prediction_stats:
        stats_dict[result] = count
    
    return render_template('respiratory_prediction.html', 
                         prediction_result=prediction_result, 
                         prediction_history=prediction_history,
                         prediction_stats=stats_dict,
                         waveform_data=waveform_data)

@app.route('/xray_prediction', methods=['GET', 'POST'])
@login_required
def xray_prediction():
    prediction_result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)
            
            # X-ray görüntüsünü yükle ve ön işleme
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img_array = np.array(img)
            
            # Gri tonlamalı görüntüyü RGB'ye dönüştür
            if len(img_array.shape) == 2:  # Eğer görüntü gri tonlamalı ise
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # Tahmin yap
            prediction = lung_diseases_model.predict(img_array)
            predicted_label = lung_diseases_numbers_map[int(np.argmax(prediction))]
            
            # Save prediction to database
            new_prediction = LungDiseasePrediction(
                user_id=current_user.id,
                filename=filename,
                prediction_result=predicted_label
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            os.remove(file_path)
            prediction_result = predicted_label
            
    # Get user's last 5 predictions
    prediction_history = LungDiseasePrediction.query.filter_by(user_id=current_user.id).order_by(LungDiseasePrediction.prediction_date.desc()).limit(5).all()
    
    # Get prediction statistics
    prediction_stats = db.session.query(
        LungDiseasePrediction.prediction_result,
        db.func.count(LungDiseasePrediction.id)
    ).filter_by(user_id=current_user.id).group_by(LungDiseasePrediction.prediction_result).all()
    
    # Convert stats to dictionary format for the chart
    stats_dict = {label: 0 for label in lung_diseases_numbers_map.values()}
    for result, count in prediction_stats:
        stats_dict[result] = count
    
    return render_template('xray_prediction.html', 
                         prediction_result=prediction_result, 
                         prediction_history=prediction_history,
                         prediction_stats=stats_dict)

@app.route('/lung_cancer_prediction', methods=['GET', 'POST'])
@login_required
def lung_cancer_prediction():
    prediction_result = None
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            uploads_dir = os.path.join(os.getcwd(), 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)
            
            # X-ray görüntüsünü yükle ve ön işleme
            img = Image.open(file_path)
            img = img.resize((200, 200))
            img_array = np.array(img)
            
            # Gri tonlamalı görüntüyü RGB'ye dönüştür
            if len(img_array.shape) == 2:  # Eğer görüntü gri tonlamalı ise
                img_array = np.stack((img_array,) * 3, axis=-1)
            
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0  # Normalize
            
            # Tahmin yap
            prediction = lung_cancer_model.predict(img_array)
            predicted_label = lung_cancer_numbers_map[int(np.argmax(prediction))]
            
            # Save prediction to database
            new_prediction = LungCancerPrediction(
                user_id=current_user.id,
                filename=filename,
                prediction_result=predicted_label
            )
            db.session.add(new_prediction)
            db.session.commit()
            
            os.remove(file_path)
            prediction_result = predicted_label
            
    # Get user's last 5 predictions
    prediction_history = LungCancerPrediction.query.filter_by(user_id=current_user.id).order_by(LungCancerPrediction.prediction_date.desc()).limit(5).all()
    
    # Get prediction statistics
    prediction_stats = db.session.query(
        LungCancerPrediction.prediction_result,
        db.func.count(LungCancerPrediction.id)
    ).filter_by(user_id=current_user.id).group_by(LungCancerPrediction.prediction_result).all()
    
    # Convert stats to dictionary format for the chart
    stats_dict = {label: 0 for label in lung_cancer_numbers_map.values()}
    for result, count in prediction_stats:
        stats_dict[result] = count
    
    return render_template('lung_cancer_prediction.html', 
                         prediction_result=prediction_result, 
                         prediction_history=prediction_history,
                         prediction_stats=stats_dict)

@app.route('/health_chatbot')
@login_required
def health_chatbot():
    return render_template('health_chatbot.html')

@app.route('/health_assistant')
@login_required
def health_assistant():
    return render_template('health_assistant.html')


# Gemini API anahtarınızı buraya ekleyin
genai.configure(api_key=api_key)

# Gemini modelini yapılandırın
model = genai.GenerativeModel('gemini-2.0-flash')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json.get('message')
        
        # Sağlık bağlamını ve kısıtlamaları ekleyin
        prompt = f"""
        Sen bir akciğer sağlığı uzmanı yapay zeka asistanısın. Sadece aşağıdaki konularda bilgi verebilirsin:
        - Konjenital Anomali
        - Mediastinal Kitle
        - Akciğer Tümörleri (İyi Huylu Tümörler, Kötü Huylu Tümörler)
        - Astım
        - KOAH
        - Bronşit
        - Akciğer kanseri
        - Akciğer iltihabı (Pnömoni)
        - Tüberküloz
        - Plevral efüzyon
        - Pnömotoraks
        - Akciğer tümörleri
        - Akciğer hastalıklarının belirtileri
        - Akciğer hastalıklarından korunma yöntemleri
        - Akciğer sağlığını koruma yolları
        - Solunum sistemi hastalıkları
        - Akciğer fonksiyon testleri
        
        Eğer soru bu konular dışında ise şu yanıtı ver:
        "Üzgünüm, ben sadece akciğer sağlığı ve hastalıkları konusunda yardımcı olabilirim. Bu konu uzmanlık alanım dışında kalıyor."
        
        Soru: {message}
        
        Not: Yanıtların bir doktor tavsiyesi yerine geçmez. Ciddi sağlık sorunları için mutlaka bir doktora başvurulmalıdır.
        """
        
        # Gemini'den yanıt alın
        response = model.generate_content(prompt)
        
        return jsonify({
            'response': response.text
        })
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

def create_pdf_report(report_text, disease_name, prediction_type, prediction_date):
    """PDF raporu oluşturan fonksiyon - ReportLab ile Türkçe karakter desteği"""
    try:
        # Türkçe karakterleri Latin harflere dönüştür
        def turkish_to_latin(text):
            replacements = {
                'ç': 'c', 'Ç': 'C',
                'ğ': 'g', 'Ğ': 'G', 
                'ı': 'i', 'I': 'I',
                'ö': 'o', 'Ö': 'O',
                'ş': 's', 'Ş': 'S',
                'ü': 'u', 'Ü': 'U'
            }
            for turkish, latin in replacements.items():
                text = text.replace(turkish, latin)
            return text
        
        # Tüm metinleri Latin harflere dönüştür
        report_text = turkish_to_latin(report_text)
        disease_name = turkish_to_latin(disease_name)
        prediction_type = turkish_to_latin(prediction_type)
        
        # PDF dosyası için geçici buffer oluştur
        buffer = io.BytesIO()
        
        # PDF dokümanı oluştur
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=72, 
            leftMargin=72, 
            topMargin=72, 
            bottomMargin=18
        )
        
        # Stil tanımlamaları
        styles = getSampleStyleSheet()
        
        # Varsayılan font olarak Helvetica kullan
        default_font = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        
        # Başlık stili
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=1,  # Ortalanmış
            textColor=colors.darkblue,
            fontName=bold_font
        )
        
        # Alt başlık stili
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=colors.darkblue,
            fontName=bold_font
        )
        
        # Normal metin stili
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            fontName=default_font
        )
        
        # PDF içeriğini oluştur
        story = []
        
        # Başlık
        title = Paragraph("ScanWiseAIht Saglik Raporu", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Tarih bilgisi
        date_text = f"Rapor Tarihi: {prediction_date.strftime('%d/%m/%Y %H:%M')}"
        date_para = Paragraph(date_text, normal_style)
        story.append(date_para)
        story.append(Spacer(1, 20))
        
        # Tahmin türü
        prediction_type_text = f"Tahmin Turu: {prediction_type}"
        prediction_type_para = Paragraph(prediction_type_text, subtitle_style)
        story.append(prediction_type_para)
        story.append(Spacer(1, 10))
        
        # Hastalık adı
        disease_text = f"Hastalık: {disease_name}"
        disease_para = Paragraph(disease_text, subtitle_style)
        story.append(disease_para)
        story.append(Spacer(1, 20))
        
        # Rapor metni
        report_para = Paragraph(report_text, normal_style)
        story.append(report_para)
        story.append(Spacer(1, 30))
        
        # Uyarı metni
        warning_text = """
        <b>Onemli Uyari:</b><br/>
        Bu rapor sadece bilgilendirme amacli olup bir doktor tavsiyesi yerine gecmez. 
        Ciddi saglik sorunlari icin mutlaka bir doktora basvurulmalidir.
        """
        warning_para = Paragraph(warning_text, normal_style)
        story.append(warning_para)
        
        # PDF'i oluştur
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        print(f"PDF oluşturma hatası: {str(e)}")
        return None

@app.route('/generate_health_report', methods=['POST'])
@login_required
def generate_health_report():
    try:
        data = request.get_json()
        disease_name = data.get('disease_name', 'Bilinmeyen Hastalık')
        prediction_type = data.get('prediction_type', 'Genel Tahmin')
        prediction_date = datetime.datetime.now()
        
        # Rapor metnini oluştur
        report_text = f"""
        Bu rapor, ScanWiseAIht yapay zeka sistemi tarafından oluşturulmuştur.
        
        Tahmin Sonucu: {disease_name}
        Tahmin Türü: {prediction_type}
        Tarih: {prediction_date.strftime('%d/%m/%Y %H:%M')}
        
        Bu rapor, akciğer sağlığı ile ilgili bir değerlendirme içermektedir. 
        Sonuçlar sadece bilgilendirme amaçlıdır ve kesin teşhis için 
        mutlaka bir doktora başvurulmalıdır.
        """
        
        # PDF oluştur
        pdf_buffer = create_pdf_report(report_text, disease_name, prediction_type, prediction_date)
        
        if pdf_buffer:
            # PDF'i base64'e çevir
            pdf_data = base64.b64encode(pdf_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'pdf_data': pdf_data,
                'filename': f'health_report_{prediction_date.strftime("%Y%m%d_%H%M%S")}.pdf'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'PDF oluşturulamadı'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download_health_report_pdf', methods=['POST'])
@login_required
def download_health_report_pdf():
    try:
        data = request.get_json()
        pdf_data = data.get('pdf_data')
        filename = data.get('filename', 'health_report.pdf')
        
        if pdf_data:
            # Base64'ten PDF'e çevir
            pdf_bytes = base64.b64decode(pdf_data)
            
            # Dosya boyutunu kontrol et
            if len(pdf_bytes) == 0:
                return jsonify({
                    'success': False,
                    'error': 'PDF dosyası boş'
                }), 400
            
            print(f"PDF indirme isteği: {filename}, boyut: {len(pdf_bytes)} bytes")
            
            # Filename'i ASCII karakterlere dönüştür ve encode et
            # Türkçe karakterleri ASCII karşılıklarına dönüştür
            filename_ascii = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode('ASCII')
            # Özel karakterleri temizle
            filename_clean = ''.join(c for c in filename_ascii if c.isalnum() or c in '._- ')
            encoded_filename = quote_header_value(filename_clean)
            
            # Response oluştur - daha iyi header'lar ile
            response = app.response_class(
                pdf_bytes,
                mimetype='application/pdf',
                headers={
                    'Content-Disposition': f'attachment; filename="{encoded_filename}"',
                    'Content-Type': 'application/pdf',
                    'Content-Length': str(len(pdf_bytes)),
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST',
                    'Access-Control-Allow-Headers': 'Content-Type'
                }
            )
            
            return response
        else:
            return jsonify({
                'success': False,
                'error': 'PDF verisi bulunamadı'
            }), 400
            
    except Exception as e:
        print(f"PDF indirme hatası: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/generate_comprehensive_health_report', methods=['POST'])
@login_required
def generate_comprehensive_health_report():
    try:
        data = request.get_json()
        disease_name = data.get('disease_name', 'Bilinmeyen Hastalık')
        prediction_type = data.get('prediction_type', 'Genel Tahmin')
        prediction_date = datetime.datetime.now()
        
        # Google API anahtarı kontrolü
        if not api_key or api_key == 'your_google_api_key_here':
            # API anahtarı yoksa basit bir rapor oluştur
            detailed_report = f"""
            {disease_name} Hastalığı Hakkında Detaylı Bilgi
            
            1. Hastalık Tanımı ve Genel Bilgiler
            {disease_name}, solunum sistemi ile ilgili bir hastalıktır. Bu hastalık, akciğerlerin normal fonksiyonlarını etkileyebilir ve çeşitli belirtilere neden olabilir.
            
            2. Belirtiler ve Semptomlar
            - Nefes darlığı
            - Öksürük
            - Göğüs ağrısı
            - Halsizlik
            - Ateş (bazı durumlarda)
            
            3. Risk Faktörleri
            - Sigara kullanımı
            - Hava kirliliği
            - Genetik faktörler
            - Yaş ve cinsiyet
            
            4. Teşhis Yöntemleri
            - Fizik muayene
            - Akciğer fonksiyon testleri
            - Görüntüleme yöntemleri (X-ray, CT)
            - Laboratuvar testleri
            
            5. Tedavi Seçenekleri
            - İlaç tedavisi
            - Solunum egzersizleri
            - Yaşam tarzı değişiklikleri
            - Gerekirse cerrahi müdahale
            
            6. Önleme ve Korunma Yöntemleri
            - Sigarayı bırakma
            - Düzenli egzersiz
            - Sağlıklı beslenme
            - Düzenli doktor kontrolü
            
            7. Yaşam Tarzı Önerileri
            - Temiz hava ortamlarında bulunma
            - Düzenli fiziksel aktivite
            - Stres yönetimi
            - Yeterli uyku
            
            8. Ne Zaman Doktora Başvurulmalı
            - Belirtilerin şiddetlenmesi
            - Yeni belirtilerin ortaya çıkması
            - Tedaviye yanıt alınamaması
            
            9. Acil Durum Belirtileri
            - Şiddetli nefes darlığı
            - Göğüs ağrısı
            - Bilinç değişiklikleri
            - Yüksek ateş
            
            10. Takip ve İzleme Önerileri
            - Düzenli doktor kontrolleri
            - Belirtilerin takibi
            - Tedavi uyumu
            - Yaşam kalitesi değerlendirmesi
            
            ÖNEMLİ: Bu rapor sadece bilgilendirme amaçlıdır ve bir doktor tavsiyesi yerine geçmez. 
            Ciddi sağlık sorunları için mutlaka bir doktora başvurulmalıdır.
            """
        else:
            # Gemini API ile detaylı rapor oluştur
            prompt = f"""
            Sen bir akciğer hastalıkları uzmanı doktorsun. Aşağıdaki hastalık için detaylı bir sağlık raporu hazırla:
            
            Hastalık: {disease_name}
            Tahmin Türü: {prediction_type}
            
            Rapor şu bölümleri içermeli:
            1. Hastalık Tanımı ve Genel Bilgiler
            2. Belirtiler ve Semptomlar
            3. Risk Faktörleri
            4. Teşhis Yöntemleri
            5. Tedavi Seçenekleri
            6. Önleme ve Korunma Yöntemleri
            7. Yaşam Tarzı Önerileri
            8. Ne Zaman Doktora Başvurulmalı
            9. Acil Durum Belirtileri
            10. Takip ve İzleme Önerileri
            
            Rapor Türkçe olmalı ve hasta dostu bir dil kullanmalı. 
            Teknik terimler açıklanmalı ve anlaşılır olmalı.
            Rapor 1000-1500 kelime civarında olmalı.
            
            Önemli: Bu rapor sadece bilgilendirme amaçlıdır ve bir doktor tavsiyesi yerine geçmez.
            """
            
            # Gemini'den detaylı rapor al
            response = model.generate_content(prompt)
            detailed_report = response.text
        
        # PDF oluştur
        pdf_buffer = create_detailed_pdf_report(detailed_report, disease_name, prediction_type, prediction_date)
        
        if pdf_buffer:
            try:
                # PDF'i base64'e çevir
                pdf_data = base64.b64encode(pdf_buffer.getvalue()).decode()
                
                # PDF boyutunu kontrol et
                pdf_size = len(pdf_data)
                print(f"PDF boyutu: {pdf_size} karakter")
                
                if pdf_size > 10000000:  # 10MB limit
                    return jsonify({
                        'success': False,
                        'error': 'PDF dosyası çok büyük'
                    }), 500
                
                return jsonify({
                    'success': True,
                    'pdf_data': pdf_data,
                    'filename': f'detailed_health_report_{disease_name.replace(" ", "_")}_{prediction_date.strftime("%Y%m%d_%H%M%S")}.pdf',
                    'report_text': detailed_report
                })
            except Exception as e:
                print(f"PDF base64 dönüştürme hatası: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': 'PDF dönüştürme hatası'
                }), 500
        else:
            return jsonify({
                'success': False,
                'error': 'PDF oluşturulamadı'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def create_detailed_pdf_report(report_text, disease_name, prediction_type, prediction_date):
    """Detaylı PDF raporu oluşturan fonksiyon"""
    try:
        # Türkçe karakterleri Latin harflere dönüştür
        def turkish_to_latin(text):
            replacements = {
                'ç': 'c', 'Ç': 'C',
                'ğ': 'g', 'Ğ': 'G', 
                'ı': 'i', 'I': 'I',
                'ö': 'o', 'Ö': 'O',
                'ş': 's', 'Ş': 'S',
                'ü': 'u', 'Ü': 'U'
            }
            for turkish, latin in replacements.items():
                text = text.replace(turkish, latin)
            return text
        
        # Tüm metinleri Latin harflere dönüştür
        report_text = turkish_to_latin(report_text)
        disease_name = turkish_to_latin(disease_name)
        prediction_type = turkish_to_latin(prediction_type)
        
        # PDF dosyası için geçici buffer oluştur
        buffer = io.BytesIO()
        
        # PDF dokümanı oluştur
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=A4, 
            rightMargin=50, 
            leftMargin=50, 
            topMargin=50, 
            bottomMargin=50
        )
        
        # Stil tanımlamaları
        styles = getSampleStyleSheet()
        
        # Varsayılan font olarak Helvetica kullan
        default_font = 'Helvetica'
        bold_font = 'Helvetica-Bold'
        
        # Başlık stili
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Ortalanmış
            textColor=colors.darkblue,
            fontName=bold_font
        )
        
        # Alt başlık stili
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=15,
            textColor=colors.darkblue,
            fontName=bold_font
        )
        
        # Normal metin stili
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            fontName=default_font
        )
        
        # Vurgu stili
        emphasis_style = ParagraphStyle(
            'CustomEmphasis',
            parent=styles['Normal'],
            fontSize=11,
            spaceAfter=10,
            fontName=bold_font,
            textColor=colors.darkred
        )
        
        # PDF içeriğini oluştur
        story = []
        
        # Başlık
        title = Paragraph("ScanWiseAI Detayli Saglik Raporu", title_style)
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Tarih ve bilgi bölümü
        info_text = f"""
        <b>Rapor Tarihi:</b> {prediction_date.strftime('%d/%m/%Y %H:%M')}<br/>
        <b>Tahmin Turu:</b> {prediction_type}<br/>
        <b>Tespit Edilen Durum:</b> {disease_name}<br/>
        <b>Rapor Olusturan:</b> ScanWiseAIht Yapay Zeka Sistemi
        """
        info_para = Paragraph(info_text, normal_style)
        story.append(info_para)
        story.append(Spacer(1, 20))
        
        # Uyarı kutusu
        warning_text = """
        <b>ONEMLI UYARI:</b><br/>
        Bu rapor sadece bilgilendirme amacli olup bir doktor tavsiyesi yerine gecmez. 
        Ciddi saglik sorunlari icin mutlaka bir doktora basvurulmalidir. 
        Bu rapor, yapay zeka destekli bir sistem tarafindan olusturulmustur.
        """
        warning_para = Paragraph(warning_text, emphasis_style)
        story.append(warning_para)
        story.append(Spacer(1, 30))
        
        # Detaylı rapor metni
        # Rapor metnini paragraflara böl
        paragraphs = report_text.split('\n\n')
        for para in paragraphs:
            if para.strip():
                # Başlık kontrolü
                if para.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                    para_style = subtitle_style
                else:
                    para_style = normal_style
                
                para_text = para.strip()
                para_para = Paragraph(para_text, para_style)
                story.append(para_para)
                story.append(Spacer(1, 10))
        
        story.append(Spacer(1, 30))
        
        # Son uyarı
        final_warning = """
        <b>SON UYARI:</b><br/>
        Bu raporu doktorunuzla paylasin ve onun onerilerini dikkate alin. 
        Kendi kendinize tedavi uygulamayin. Acil durumlarda 112'yi arayin.
        """
        final_warning_para = Paragraph(final_warning, emphasis_style)
        story.append(final_warning_para)
        
        # PDF'i oluştur
        doc.build(story)
        buffer.seek(0)
        
        return buffer
        
    except Exception as e:
        print(f"Detaylı PDF oluşturma hatası: {str(e)}")
        return None

@app.route('/get_disease_info', methods=['POST'])
@login_required
def get_disease_info():
    """Hastalık hakkında kısa bilgi döndüren endpoint"""
    try:
        data = request.get_json()
        disease_name = data.get('disease_name', '')
        
        # Google API anahtarı kontrolü
        if not api_key or api_key == 'your_google_api_key_here':
            # API anahtarı yoksa basit bilgi döndür
            disease_info = f"""
            {disease_name} hastalığı, solunum sistemi ile ilgili bir durumdur. 
            Bu hastalık genellikle nefes darlığı, öksürük ve göğüs ağrısı gibi belirtilerle kendini gösterir. 
            Risk faktörleri arasında sigara kullanımı, hava kirliliği ve genetik faktörler bulunur. 
            Erken teşhis ve tedavi önemlidir. Belirtileriniz varsa mutlaka bir doktora başvurun.
            """
        else:
            # Gemini API ile kısa bilgi al
            prompt = f"""
            {disease_name} hastalığı hakkında kısa ve öz bilgi ver (100-150 kelime). 
            Belirtiler, risk faktörleri ve genel bilgileri içersin.
            Türkçe olmalı ve hasta dostu bir dil kullanmalı.
            """
            
            response = model.generate_content(prompt)
            disease_info = response.text
        
        return jsonify({
            'success': True,
            'disease_info': disease_info
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)