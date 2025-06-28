from torchinfo import summary
from models.small_mobilenet import SmallMobileNet


summary(SmallMobileNet(), input_size=(1, 1, 28, 28))
