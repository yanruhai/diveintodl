def text_labels(indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',#pullover 套衫
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']#sandal:凉鞋 ankleboot:踝鞋
    return [labels[i] for i in indices]

print(text_labels(range(1,3)))