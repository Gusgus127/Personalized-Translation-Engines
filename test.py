import transformers
try:
    transformers.MarianMTModel.from_pretrained("Gusgus127/marianmt-es-en-medical")
except Exception as e:
    print("--- ERROR LOG START ---")
    print(e)
    print("--- ERROR LOG END ---")