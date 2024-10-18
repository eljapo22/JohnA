from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('layout/', include('layout.urls')),  # Include the layout app URLs
    path('', include('layout.urls')),  # Redirect the root URL to layout app
]
