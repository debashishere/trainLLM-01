



def main():
    print("Executiing Main------------")
    text = None
    text = open("./data/shakespeare.txt").read()
    if (text and type(text) == str):
        print("Valid Text Detected with len------------", len(text))
        chars = sorted(set(text))
        print("Number of unique characters------------", len(chars))
        itos = {i: c for i, c in enumerate(chars)}
        stoi = {c: i for i, c in enumerate(chars)}
        
    else:
        print("Invalid Text Input.")

    def encode(text):
        return [ stoi[char] for char in text ]

    def decode(ids):
        return [ itos[char] for char in ids]

    encodedText = encode("Hello")
    print("Encoded Text------------", encodedText)
    decodedText = decode(encodedText)
    print("Decoded Text------------", decodedText)

    print("END OF -----Read text ------------")
    
    



if __name__ == "__main__":
    main()