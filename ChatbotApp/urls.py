from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
               path("UserLogin.html", views.UserLogin, name="UserLogin"),	      
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
	       path("Register.html", views.Register, name="Register"),	      
               path("RegisterAction", views.RegisterAction, name="RegisterAction"),
               path("RunRandom", views.RunRandom, name="RunRandom"),
               path("RunDT.html", views.RunDT, name="RunDT"),
               path("RunSVM", views.RunSVM, name="RunSVM"),
               path("Chatbot", views.Chatbot, name="Chatbot"),           
	       path("ChatData", views.ChatData, name="ChatData"), 
]
