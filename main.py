import logging
logging.basicConfig(filename='app.log', filemode='w', level=logging.DEBUG)

from flair.models import SequenceTagger
from flair.data import Sentence
from ucscenario.src.api.utils.diagram_generator_api import *
from dfd_to_padfd import generate_pa_dfd_xml

import spacy
import os
import csv
import networkx as nx
import string
import json
from shutil import copyfile
from so_that import process_so_that, split_us
from similarity_util import get_similarity

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import collections

class StoryDFD():
    def __init__(self):
        self.initModels()
        self.privacy_only = False
        self.so_that = {}
        self.root_folder = "dfd_output/"

    def initModels(self):
        logging.info('Initating NER Model')
        NER_MODEL_PATH = "ner-model.pt"
        self.model = SequenceTagger.load(NER_MODEL_PATH)

        logging.info('Initating Spacy NLP Model')
        self.nlp = spacy.load("en_core_web_sm")


    def setStories(self, stories):
        self.stories = stories

    def setStoriesFromFile(self, filename):
        self.stories = []

        # some project are error due to the encoding
        # experiment with several encoding

        # with open(filename, 'r', encoding='cp1252') as f:
        # with open(filename, 'r', encoding='utf-8') as f:

        with open(filename, 'r', encoding='utf-8-sig') as f:
            for story in f:
                self.stories.append(story)

    def writeFilteredStoriesToFile(self, filename=None):
        stories_tagged = []
        for data in self.ner_data:
            new_obj = {}
            if data["text"] in self.filtered_stories:
                new_obj["text"] = data["text"]
                new_obj["entities"] = []

                for entity in data["entities"]:
                    new_obj["entities"].append({
                        "text" : entity["text"],
                        "label" : entity["labels"][0].value
                    })


                stories_tagged.append(new_obj)

        if filename is None:
            filename = "stories_processed"

        with open(filename + ".json", "w") as outfile:
            json.dump(stories_tagged, outfile, indent=2)

        
    def getUniqueEntitiesByLabel(self, entity_label):
        entities = []
        entities_dict = {}

        for data in self.ner_data:
            for entity in data["entities"]:
                label = entity["labels"][0].value
                if entity_label != label:
                    continue

                if entity["text"] not in entities_dict:
                    entities.append(entity["text"])
                    entities_dict[entity["text"]] = [data["text"]]
                else:
                    entities_dict[entity["text"]].append(data["text"])

        # key = entity text, value = user story text
        return entities_dict, entities

    def getStoriesByLabel(self, entity_label):
        stories = []

        for data in self.ner_data:
            for entity in data["entities"]:
                label = entity["labels"][0].value
                if entity_label != label:
                    continue

                stories.append(data["text"])

        return stories

    def generateRobustnessDiagram(self):
        result = None
        if len(self.filtered_stories) == 1:
            result = DiagramGenerator.generate(self.filtered_stories[0])
        elif len(self.filtered_stories) > 1:
            result = DiagramGenerator.multi(self.filtered_stories,True)
        else:
            print("No user stories to generate")

        return result

    def generateSoThatDict(self, story):            
        actors, verb, personal_data = process_so_that(story)
        if personal_data:
            first, second, third, idx_cut_first, idx_cut_second = split_us(story)

            self.so_that[story] = {
                "first" : first,
                "second" : second,
                "third" : third,
                "actor" : actors,
                "verb" : verb,
                "personal_data" : personal_data,
            }

    def removePunctuation(self, s):
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in s if ch not in exclude)

        return s

    def buildNER(self):
        logging.info('Inferring NER')
        self.ner_data = []
        for i, story in enumerate(self.stories):
            sentence = Sentence(story)
            self.model.predict(sentence)
            self.ner_data.append(sentence.to_dict(tag_type='ner'))

    def buildNLP(self):
        logging.info('Building NLP data from the Stories')

        self.nlp_data = []
        for i, story in enumerate(self.stories):
            doc = self.nlp(story)
            self.nlp_data.append(doc)

    
    def preprocessWord(self, word):
        word = self.removePunctuation(word.lower())
        word = lemmatizer.lemmatize(word).capitalize().strip()

        return word

    def processDFDPerStory(self, folder_output, unify_dfd=False):
        self.buildNER()

        dfd_folder = self.root_folder + folder_output + "/"
        _, self.personal_data_entities = self.getUniqueEntitiesByLabel("PII")
        
        if not os.path.exists(dfd_folder):
            os.mkdir(dfd_folder)

        error_story = []
        error_cause = []
        if not unify_dfd:
            for i, story in enumerate(self.stories):
                dfd_output_name_num = dfd_folder + "s_{}".format(i+1)

                if os.path.exists(dfd_output_name_num + '.xml'):
                    continue

                try:
                    result = DiagramGenerator.generate(story)

                    self.so_that = {}
                    self.generateSoThatDict(story.replace(".",""))

                    print("**"*10)
                    print(story)
                    print(self.so_that)
                    print("**"*10)

                    if self.so_that:
                        with open(dfd_output_name_num + "_so.txt", 'w') as ff:
                            ff.write(str(self.so_that))

                    self.robustToDFD(result["id"], dfd_output_name_num)
                    
                except Exception as e:
                    error_story.append(story)
                    error_cause.append(str(e))

        else:
            self.filtered_stories = self.stories
            self.so_that = {}
            for story in self.stories:
                self.generateSoThatDict(story)


            robustness_result = self.generateRobustnessDiagram()

            # READ THE OUTPUT OF ROBUSTNESS DIAGRAM and TRANSFORM it to DFD
            self.robustToDFD(robustness_result["id"], dfd_folder + "stories")

        with open("error_story.txt", 'w') as f:
            for i, s in enumerate(error_story):
                f.write(s)
                f.write(error_cause[i])
                f.write("=="*10)



    def processDFDPerEntity(self, process_all=False, us_name=None, based_on="data_subject", force_single=False):
        self.buildNER()
        self.buildNLP()

        Entity_Based_Stories, Personal_Data_Stories, Processing_Stories = {}, {}, {}

        if based_on == "data_subject":
            Entity_Based_Stories, _ = self.getUniqueEntitiesByLabel("Data")
            _, self.personal_data_entities = self.getUniqueEntitiesByLabel("PII")

            Personal_Data_Stories = self.getStoriesByLabel("PII")
            Processing_Stories = self.getStoriesByLabel("Processing")
        else:
            Entity_Based_Stories, self.personal_data_entities = self.getUniqueEntitiesByLabel("PII")
            Processing_Stories = self.getStoriesByLabel("Processing")

        Entity_stem = {}
        for entity, stories in Entity_Based_Stories.items():
            filtered_stories_privacy_only = []
            if self.privacy_only:
                if based_on == "data_subject":
                    filtered_stories_privacy_only = [story for story in stories if story in Personal_Data_Stories or story in Processing_Stories]
                else:
                    filtered_stories_privacy_only = [story for story in stories]

            stemmed_word = self.preprocessWord(entity)
            processed_stories = filtered_stories_privacy_only if self.privacy_only else stories

            if len(processed_stories) == 0:
                continue

            if stemmed_word in Entity_stem:
                Entity_stem[stemmed_word].extend(processed_stories)
            else:
                Entity_stem[stemmed_word] = processed_stories

        Entity_ordered = collections.OrderedDict(sorted(Entity_stem.items()))
        
        entity_name = "Data Subject" if based_on == "data_subject" else "Personal Data"
        
        if not process_all:
            print("Choose {} to start from: ".format(entity_name))
            subj_map = []
            counter = 1
            for entity, stories in Entity_ordered.items():
                print(counter, entity)
                counter += 1

                subj_map.append(entity)

            counter_input = int(input("Type in the {} number = ".format(entity_name)))

            self.filtered_stories = Entity_stem[subj_map[counter_input-1]]
            self.writeFilteredStoriesToFile(filename=subj_map[counter_input-1])

            # GENERATE ROBUSTNESS DIAGRAM
            robustness_result = self.generateRobustnessDiagram()

            # READ THE OUTPUT OF ROBUSTNESS DIAGRAM and TRANSFORM it to DFD
            self.robustToDFD(robustness_result["id"])
        else:
            for data_subject, stories in Entity_ordered.items():
                print("Processing ", data_subject)

                self.filtered_stories = stories

                #try:
                
                # READ THE OUTPUT OF ROBUSTNESS DIAGRAM and TRANSFORM it to DFD
                dfd_folder = self.root_folder + us_name
                if not os.path.exists(dfd_folder):
                    os.mkdir(dfd_folder)

                dfd_output_name = dfd_folder + "/" + data_subject
                self.writeFilteredStoriesToFile(filename=dfd_output_name)

                if os.path.exists(dfd_output_name + '.xml'):
                    continue
                
                # GENERATE ROBUSTNESS DIAGRAM
                if force_single:
                    for i, story in enumerate(stories):
                        dfd_output_name_num = dfd_output_name + "_{}".format(i+1)
                        result = DiagramGenerator.generate(story)

                        self.so_that = {}
                        self.generateSoThatDict(story)

                        if self.so_that:
                            with open(dfd_output_name_num + "_so_that.txt", 'w') as ff:
                                ff.write(str(self.so_that))
                        
                        self.robustToDFD(result["id"], dfd_output_name_num)

                else:
                    robustness_result = self.generateRobustnessDiagram()

                    self.so_that = {}
                    for story in self.filtered_stories:
                        self.generateSoThatDict(story)

                    self.robustToDFD(robustness_result["id"], dfd_output_name)
                
                #except:
                #    pass



    def generateDfdGraphiz(self, dot_rows, dfd_output_name):
        with open(dfd_output_name + ".dot", 'w') as dfd:
            dfd.write("digraph dfd2{ \n")

            for dot in dot_rows:
                dfd.write(dot + "\n")

            dfd.write("}")

        os.system("dot -Tpng \"{}.dot\" -o \"{}_dfd.png\"".format(dfd_output_name, dfd_output_name))


    def robustToDFD(self, ROBUST_FILE, dfd_output_name=None):
        # DRAW IO STYLE

        DFD_CSV_HEADER = ["id","value","style","source","target","type"]
        DFD_PROP = {
            'external_entity' : {
                'style' : 'rounded=0',
                'type' : 'external_entity'
            }, 
            'process' : {
                'style' : 'ellipse',
                'type' : 'process'
            }, 
            'dataflow' : {
                'style' : 'endArrow=classic',
                'type' : 'endArrow=classic'
            }, 
            'datastore' : {
                'style' : 'shape=partialRectangle',
                'type' : 'data_base'
            }, 
        }

        FILE_ROBUST = "ucscenario/src/api/out/multi/" + ROBUST_FILE + ".txt"
        if not os.path.isfile(FILE_ROBUST):
            FILE_ROBUST = "ucscenario/src/api/out/single/" + ROBUST_FILE + ".txt"

        copyfile(FILE_ROBUST, dfd_output_name + "_robust.txt")
        copyfile(FILE_ROBUST.replace(".txt", ".png"), dfd_output_name + "_robust.png")

        ext_entity = {}
        process = {}
        boundary = {}
        entity = {}
        id_dfd = 200
        id_process = 1
        dfd_rows = []
        dot_rows = ["node[shape=record]"]
        G = nx.Graph()
        
        # MAKE A MAPPING FROM ROBUSNESS DIAGRAM FILE
        with open(FILE_ROBUST, 'r') as ff:
            
            for f in ff:
                uml_part = f.split()

                # external entity is actor, connected to process
                if len(uml_part) >= 4:
                    if uml_part[0] == 'actor' and 'as' in uml_part:
                        alias_index = uml_part.index('as') + 1
                        ext_entity[uml_part[alias_index]] = {}

                        ext_entity_label =  ' '.join(uml_part[1:alias_index-1]).replace('"','').replace('\\n',' ')

                        ext_entity[uml_part[alias_index]]['label'] = ext_entity_label
                        ext_entity[uml_part[alias_index]]['process'] = []
                        ext_entity[uml_part[alias_index]]['id_dfd'] = id_dfd

                        row = [id_dfd, ext_entity_label, DFD_PROP['external_entity']['style'], "null", "null", DFD_PROP['external_entity']['type']]

                        dfd_rows.append({"id":row[0], "data":row})
                        dot_rows.append("{} [label=\"{}\" shape=box];".format(row[0], ext_entity_label))
                        id_dfd += 1


                    # boundary act as a bridge from actor to process
                    if uml_part[0] == 'boundary' and 'as' in uml_part:
                        alias_index = uml_part.index('as') + 1
                        boundary[uml_part[alias_index]] = {}

                        boundary[uml_part[alias_index]]['label'] = ' '.join(uml_part[1:alias_index-1]).replace('"','').replace('\\n',' ')
                        boundary[uml_part[alias_index]]['actor'] = []

                    # control is process
                    if uml_part[0] == 'control' and 'as' in uml_part:
                        alias_index = uml_part.index('as') + 1
                        process[uml_part[alias_index]] = {}

                        process[uml_part[alias_index]]['label'] = ' '.join(uml_part[1:alias_index-1]).replace('"','').replace('\\n',' ')
                        process[uml_part[alias_index]]['process'] = []
                        process[uml_part[alias_index]]['entity'] = []
                        process[uml_part[alias_index]]['id_dfd'] = id_dfd
                        
                        row = [id_dfd, process[uml_part[alias_index]]['label'], DFD_PROP['process']['style'], "null", "null", DFD_PROP['process']['type']]

                        dfd_rows.append({"id":row[0], "data":row})
                        dot_rows.append("{} [label=\"{{<f0> {}.0|<f1> {} }}\" shape=Mrecord];".format(row[0], id_process, process[uml_part[alias_index]]['label']))
                        id_dfd += 1
                        id_process += 1


                    # entity can become personal data
                    if uml_part[0] == 'entity' and 'as' in uml_part:
                        alias_index = uml_part.index('as') + 1
                        entity_label_dfd = ' '.join(uml_part[1:alias_index-1]).replace('"','').replace('\n',' ')

                        if self.privacy_only:
                            if entity_label_dfd.lower().strip() not in (lemmatizer.lemmatize(pdata).lower().strip() for pdata in self.personal_data_entities):
                                continue

                        entity[uml_part[alias_index]] = {}

                        entity[uml_part[alias_index]]['label'] = entity_label_dfd
                        entity[uml_part[alias_index]]['id_dfd'] = id_dfd
                        entity[uml_part[alias_index]]['entity'] = []
                        
                        row = [id_dfd, entity[uml_part[alias_index]]['label'], DFD_PROP['datastore']['style'], "null", "null", DFD_PROP['datastore']['type']]

                        dfd_rows.append({"id":row[0], "data":row})

                        # make red color for PII
                        color = ""
                        for pdata in self.personal_data_entities:
                            if entity_label_dfd.lower().strip() in pdata.lower() or pdata.lower() in entity_label_dfd.lower().strip():
                                color = "color=red"
                                break

                        dot_rows.append("{} [label=\"<f0>  |<f1> {} \" {}];".format(row[0], entity[uml_part[alias_index]]['label'], color))
                        
                        id_dfd += 1


                # the dependency between element
                if len(uml_part) == 3 and (uml_part[-1] in ext_entity or uml_part[-1] in process or uml_part[-1] in boundary or uml_part[-1] in entity):

                    G.add_edge(uml_part[0], uml_part[-1])
                
                    # the connection between actor and boundary must appear first
                    if uml_part[0] in ext_entity:
                        boundary[uml_part[-1]]['actor'].append(uml_part[0])

                    # map actor to process via boundary
                    if uml_part[0] in boundary:
                        for actor in boundary[uml_part[0]]['actor']:
                            ext_entity[actor]['process'].append(uml_part[-1])

                    # map process to another process
                    if uml_part[0] in process:
                        process[uml_part[0]]['process'].append(uml_part[-1])

                    # map entity to process
                    if uml_part[0] in entity:
                        if uml_part[-1] in process:
                            process[uml_part[-1]]['entity'].append(uml_part[0])
                        else:
                            entity[uml_part[-1]]['entity'].append(uml_part[0])

        # END OF MAPPING

        # SO THAT
        if self.so_that:
            id_dfd_so = 1000

            for story, so_data in self.so_that.items():
                # processing is always new, start with processing, because it's the center
                row = [id_process, so_data["verb"], DFD_PROP['process']['style'], "null", "null", DFD_PROP['process']['type']]
                dfd_rows.append({"id":row[0], "data":row})
                dot_rows.append("{} [label=\"{{<f0> {}.0|<f1> {} }}\" shape=Mrecord];".format(row[0], id_process, so_data["verb"]))

                PROCESS_CONNECTED = False

                for id_actor, data_actor in ext_entity.items():
                    for actor_so in so_data['actor']:
                        if get_similarity(actor_so, data_actor['label']) < 0.8:
                            # if we found a new actor in so that, we add it to DFD
                            row = [id_dfd_so, actor_so.capitalize(), DFD_PROP['external_entity']['style'], "null", "null", DFD_PROP['external_entity']['type']]

                            dfd_rows.append({"id":row[0], "data":row})
                            dot_rows.append("{} [label=\"{}\" shape=box];".format(row[0], actor_so.capitalize()))
                            
                            id_dfd_so += 1

                            # directly make the connection
                            row = [id_dfd_so, "", DFD_PROP['dataflow']['style'], id_dfd_so - 1, id_process, DFD_PROP['dataflow']['type']]
                            dfd_rows.append({"id":row[0], "data":row})
                            dot_rows.append("{} -> {}".format(id_dfd_so - 1, id_process))

                            id_dfd_so += 1
                        else:
                            # IF IT IS OLD ACTOR; THE ARROW SHOULD COME FROM THE PROCESSING IN THE SECOND PART OF USER STORY
                            for proc in data_actor['process']:
                                if (process[proc]['label'].strip().lower() in so_data["second"].strip().lower() or get_similarity(process[proc]['label'], so_data["second"]) > 0.7) and not PROCESS_CONNECTED:
                                    # make the connection from process in the second part of user story
                                    PROCESS_CONNECTED = True
                                    row = [id_dfd_so, "", DFD_PROP['dataflow']['style'], process[proc]['id_dfd'], id_process, DFD_PROP['dataflow']['type']]
                                    dfd_rows.append({"id":row[0], "data":row})
                                    dot_rows.append("{} -> {}".format(process[proc]['id_dfd'], id_process))

                                    id_dfd_so += 1


                            # If No Data Flow connected to our process, then connect it to the actor
                            if not PROCESS_CONNECTED:
                                row = [id_dfd_so, "", DFD_PROP['dataflow']['style'], data_actor["id_dfd"], id_process, DFD_PROP['dataflow']['type']]
                                dfd_rows.append({"id":row[0], "data":row})
                                dot_rows.append("{} -> {}".format(data_actor["id_dfd"], id_process))

                                id_dfd_so += 1

                # personal data can be new
                for personal_data_so in so_data['personal_data']:
                    NEW_ENTITY = True
                    for id_entity, data_entity in entity.items():
                        if get_similarity(personal_data_so, data_entity['label']) > 0.8:
                            NEW_ENTITY = False

                            # directly make the connection
                            row = [id_dfd_so, personal_data_so, DFD_PROP['dataflow']['style'], id_process, data_entity["id_dfd"], DFD_PROP['dataflow']['type']]
                            dfd_rows.append({"id":row[0], "data":row})
                            dot_rows.append("{} -> {}  [label=\"{}\"]".format(id_process, data_entity["id_dfd"], personal_data_so))

                            id_dfd_so += 1

                        
                    # if we found a new personal data in so that, we add it to DFD
                    if NEW_ENTITY:
                        row = [id_dfd_so, personal_data_so, DFD_PROP['external_entity']['style'], "null", "null", DFD_PROP['external_entity']['type']]

                        dfd_rows.append({"id":row[0], "data":row})
                        dot_rows.append("{} [label=\"<f0>  |<f1> {} \" {}];".format(row[0], personal_data_so, "color=red"))
                        
                        id_dfd_so += 1

                        # directly make the connection
                        row = [id_dfd_so, personal_data_so, DFD_PROP['dataflow']['style'], id_process, id_dfd_so - 1, DFD_PROP['dataflow']['type']]
                        dfd_rows.append({"id":row[0], "data":row})
                        dot_rows.append("{} -> {} [label=\"{}\"]".format(id_process, id_dfd_so - 1, personal_data_so))

                        id_dfd_so += 1
                
                id_process += 1

        # END OF SO THAT

        # GENERATE DFD

        for k,v in ext_entity.items():
            # connect actor and process
            for proc in v['process']:
                # arrow connection is different element
                # i'm still not sure about the label between actor and process
                id_dfd += 1
                row = [id_dfd, "", DFD_PROP['dataflow']['style'], v['id_dfd'], process[proc]['id_dfd'], DFD_PROP['dataflow']['type']]
                dfd_rows.append({"id":row[0], "data":row})
                dot_rows.append("{} -> {}".format(v['id_dfd'], process[proc]['id_dfd']))

        mapped_entity = []
        for k,v in process.items():
            # arrow connection between process
            # i'm still not sure about the label between process
            for proc in v['process']:
                id_dfd += 1
                row = [id_dfd, "", DFD_PROP['dataflow']['style'], v['id_dfd'], process[proc]['id_dfd'], DFD_PROP['dataflow']['type']]
                dfd_rows.append({"id":row[0], "data":row})
                dot_rows.append("{} -> {}".format(v['id_dfd'], process[proc]['id_dfd']))

            # arrow connection between process and data store
            for ent in v['entity']:
                id_dfd += 1
                row = [id_dfd, entity[ent]['label'], DFD_PROP['dataflow']['style'], v['id_dfd'], entity[ent]['id_dfd'], DFD_PROP['dataflow']['type']]
                dfd_rows.append({"id":row[0], "data":row})
                dot_rows.append("{} -> {} [label=\"{}\"]".format(v['id_dfd'], entity[ent]['id_dfd'], entity[ent]['label']))

                mapped_entity.append(entity[ent]['label'])

        # how to map entity to entity?
        # There is no connection between data store in DFD
        # Best way is to connect the entity with the process, based on the parent entity
        for k,v in entity.items():
            if v['label'] in mapped_entity:
                continue

            for kp,vp in process.items():
                id_dfd += 1

                if v["label"].strip().lower() in vp["label"].strip().lower():
                    row = [id_dfd, v['label'], DFD_PROP['dataflow']['style'], vp['id_dfd'], v['id_dfd'], DFD_PROP['dataflow']['type']]
                    dfd_rows.append({"id":row[0], "data":row})
                    dot_rows.append("{} -> {} [label=\"{}\"]".format(vp['id_dfd'], v['id_dfd'], v['label']))

                    break



        dfd_rows.sort(key=lambda x: x.get('id'))
        sorted_dfd_rows = [v["data"] for v in dfd_rows]
        sorted_dfd_rows.insert(0, DFD_CSV_HEADER)

        self.generateDfdGraphiz(dot_rows, dfd_output_name)

        with open(dfd_output_name + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sorted_dfd_rows)
        

        pa_dfd_xml=generate_pa_dfd_xml(dfd_output_name + '.csv',dfd_output_name + '.xml')

        # END GENERATE DFD
    


dd = StoryDFD()

dd.setStoriesFromFile("g12-camperplus.txt")

# process per user story
dd.processDFDPerStory("g12-camperplus")

# process per entity, default process per data subject, see the parameter for more
# dd.processDFDPerEntity(process_all=True, us_name="camperplus_gabung", force_single=False)