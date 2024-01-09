import spacy
from flair.data import Sentence
from flair.models import SequenceTagger

class SoThatProcessor:
    def __init__(self, ner_model, nlp_model):
        self.nlp = nlp_model
        self.ner_model = ner_model
        
    def split_us(self, us):
        """
        us: String that respects the format "AS ... ,I ... (SO THAT ...)"
        return ["AS part","I WANT part","SO THAT part"]
        """
        idx_cut = us.lower().find("so that")

        idx_so_that = idx_cut
        
        if(idx_cut == -1):
            third = ""
            other = us
        else:  
            third = us[idx_cut:]
            other = us[:idx_cut]
     
        idx_cut = other.lower().find(", i ")
        if idx_cut == -1:
            idx_cut = other.lower().find(", i'd")
        add = 2
        if idx_cut == -1:
            idx_cut = other.lower().find(", i'm")
        add = 2
        if(idx_cut == -1):
            idx_cut = other.lower().find(",i ")
            if idx_cut == -1:
                idx_cut = other.lower().find(",i'd")
            add = 1
            
            if(idx_cut == -1):
                idx_cut = other.lower().find(" i ")
                if idx_cut == -1:
                    idx_cut = other.lower().find(" i'd")
                add = 0
        first = other[:idx_cut+1]    
        second = other[idx_cut + add:]
        
        if(first.lower().find("as ") == -1):
            first = ""
        
        return first.strip(),second.strip(),third.strip(), idx_cut, idx_so_that

    def get_compounds(self, doc):
        compounds = [tok for tok in doc if tok.dep_ == 'compound'] # Get list of compounds in doc
        compounds = [c for c in compounds if c.i == 0 or doc[c.i - 1].dep_ != 'compound'] # Remove middle parts of compound nouns, but avoid index errors

        noun_list = []
        if compounds: 
            for tok in compounds:
                noun = doc[tok.i: tok.head.i + 1]
                pair_item_1 = noun
                
                if noun.root.dep_ in ['nsubj', 'nsubjpass']:
                    noun_list.append(str(pair_item_1))

        if len(noun_list) == 0:
            noun_list = [tok.text for tok in doc if tok.dep_ == 'nsubj']

        return noun_list

    
    def parse_so_that(self, third, processing, first_actor, second_actor):
        doc = self.nlp(third)

        if third.lower().startswith("so that "):
            third = third[len("so that"):].strip()
        
        token_processing_index = 0
        if processing is None:
            for token in doc:
                if token.pos_ == "VERB" and token.tag_ == "VB":
                    processing = str(doc[token.i:])
                    token_processing_index = token.i

        # seach for negation
        # if negation appear before main VERB, just return None 
        # it is usually a non-functional requirement, which are not suitable for DFD
        for token in doc:
            if token.dep_ == "neg" and token.i < token_processing_index:
                return None, None

        if processing is not None:
            third = third.replace(processing.strip(),"")

        actors = []

        if third.lower().startswith("they"):
            actors = [second_actor]
        elif third.lower().startswith("i ") or third.lower().startswith("i'm ") or third.lower().startswith("my "):
            actors = [first_actor]
        else:
            actors = self.get_compounds(doc)
            if 'I' in actors or 'it' in actors:
                actors = [first_actor]

        '''
        debug purposes
        for i, token in enumerate(doc):
            print(i, token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)

        '''

        return actors, processing
    
    def process_so_that(self, us):
        first, second, third, idx_cut_first, idx_cut_second = self.split_us(us)

        

        

        # print(first, "2", second, "3" , third)

        # we want to know the actor that have a role in "so that"
        # three possible actor:
        # I, I'm, my -> actor in first part
        # they -> actor in second part
        # Direct Actor, can be identified directly in so that part
        first_actor = None
        second_actor = None

        # if there are any personal data involved in so that part, we parse it
        # also, search for the first Processing Entity appear in "so that" part
        personal_data_in_so_that = []
        processing_idx_in_so_that = None
        

        if len(third) > 0:
            sentence = Sentence(us)
            self.ner_model.predict(sentence)
        
            sentence.to_dict(tag_type='ner')
            
            for entity in sentence.get_spans('ner'):
                
                # check first actor
                if entity.get_label("ner").value == "Data" and entity.start_position < idx_cut_first:
                    first_actor = entity.text

                # check second actor
                if second_actor is None:
                    if entity.start_position > idx_cut_first and entity.start_position < idx_cut_second:
                        if entity.get_label("ner").score > 0.5 and entity.get_label("ner").value == "Data":
                            second_actor = entity.text

                # get personal data involved
                if entity.get_label("ner").value == "PII" and entity.start_position > idx_cut_second:
                    personal_data_in_so_that.append(entity.text.capitalize())

                # search for the first processing entity
                '''
                if entity["labels"][0].value == "Processing" and entity["start_pos"] > idx_cut_second:
                    if processing_idx_in_so_that is None:
                        processing_idx_in_so_that = entity["start_pos"]
                    elif entity["start_pos"] < processing_idx_in_so_that:
                        processing_idx_in_so_that = entity["start_pos"]
                '''

        if len(personal_data_in_so_that) > 0:
            # actor may be more than one
            # processing my be indicated by NER model
            # if not index found
            
            processing = None
            if processing_idx_in_so_that:
                processing = us[processing_idx_in_so_that:]

            if first_actor is None:
                first_actor = self.get_compounds(self.nlp(first))

            if second_actor is None:
                second_actor = self.get_compounds(self.nlp(second))

            actors, verb = self.parse_so_that(third, processing, first_actor, second_actor)

            # if there are no verb / processing, we don't need to process it further to DFD
            if verb is not None and len(actors) > 0:
                return actors, verb.capitalize(), personal_data_in_so_that

        return None, None, None