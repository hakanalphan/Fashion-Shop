from django.shortcuts import render
from .models import Product
import numpy as np
import pickle
import os
from django.conf import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow as tf


# Create your views here.
def index(request):
    products = Product.objects.all()
    context = {'products':products}
    return render(request, 'index.html', context)



from django.shortcuts import render
from django.conf import settings
from .forms import UploadImageForm
from .utils import save_uploaded_file, feature_extraction, content_based_recommendation
from .models import UploadedImage
import numpy as np

# Önceden yüklenmiş model ve veri yapısının yüklenmesi
feature_list = np.array(pickle.load(open(os.path.join(settings.BASE_DIR, 'FashionShop/static/featurevector.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join(settings.BASE_DIR, 'filename.pkl'), 'rb'))
image_names = [os.path.basename(file) for file in filenames]

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def fashion_recommender(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['image']
            if save_uploaded_file(uploaded_file):
                # Resimden özellik çıkarma
                features = feature_extraction(os.path.join(settings.BASE_DIR, 'uploads', uploaded_file.name), model)
                # İçerik tabanlı öneri
                recommended_images_content_based = content_based_recommendation(features, feature_list, image_names)
                
                # Yüklenen resmi veritabanına kaydetme
                uploaded_image = UploadedImage(image=uploaded_file)
                uploaded_image.save()
                
                return render(request, 'recommendations.html', {'uploaded_file': uploaded_file, 'recommended_images_content_based': recommended_images_content_based})
    else:
        form = UploadImageForm()
    return render(request, 'upload.html', {'form': form})

