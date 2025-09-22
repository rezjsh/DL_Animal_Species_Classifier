from app import create_app

app = create_app(model_choice='EfficientNetB0')
port = 5000
if __name__ == '__main__':
    app.run(debug=True, port=port)
