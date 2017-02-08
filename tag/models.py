from __future__ import unicode_literals

from django.db import models
from colorful.fields import RGBColorField
import random

# utility function
def getHexColor():
	r = lambda: random.randint(0,255)
	return ('#%02X%02X%02X' % (r(),r(),r()))



class Tag(models.Model):
	tag_color=RGBColorField(default=getHexColor)	

