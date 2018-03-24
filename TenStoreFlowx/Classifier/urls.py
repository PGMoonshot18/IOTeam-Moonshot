from django.conf.urls import url
from . import views

urlpatterns = [
	url(r'^getLiveFeed$', views.getLatestImage),
	url(r'^getLiveFeed/$', views.getLatestImage),
	url(r'^toggleCapture$', views.toggleCapture),
	url(r'^toggleCapture/$', views.toggleCapture),
	url(r'^$', views.defaultPage, name='defaultPage'),
	url(r'^', views.defaultPage, name='defaultPage'),
]