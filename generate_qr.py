import qrcode

# attendance system link
url = "http://192.168.1.16:5000"

qr = qrcode.make(url)

qr.save("attendance_qr.png")

print("QR Code generated successfully!")