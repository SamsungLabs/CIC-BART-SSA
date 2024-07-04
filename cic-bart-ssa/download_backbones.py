
from transformers import BartForConditionalGeneration, BartTokenizer

if __name__ == '__main__':

    print('Downloading checkpoints if not cached')

    print('BART-base')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    print('Done!')

