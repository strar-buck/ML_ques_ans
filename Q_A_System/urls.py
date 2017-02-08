from django.conf.urls import url,include
from django.contrib import admin
from django.conf import settings

urlpatterns = [
    url(r'^admin/', include(admin.site.urls)),
    url(r'^$','tag.views.home',name='home'),

    url(r'^predict_tag/$','tag.views.predict_tag',name='predict_tag'),

    url(r'^predict_sentiment/$','tag.views.predict_sentiment',name='predict_sentiment'),
]

urlpatterns +=[
        url(r'^static/(?P<path>.*)$', 'django.views.static.serve', {'document_root': settings.STATIC_ROOT}),
]
