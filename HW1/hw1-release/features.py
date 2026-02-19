from collections import Counter
from data import Example

class FeatureExtractor:
    def extract_features(self, sentence: list[str]) -> Counter:
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, max_num_features: int = 500):
        self.max_num_features = max_num_features
        self.vocabulary = set([])

    def build_vocabulary(self, examples: list[Example]):
        cnt = Counter()
        for example in examples:
          words = example.get_combined_words()
          cnt.update(words)
          
        for word, _ in cnt.most_common(self.max_num_features):
          self.vocabulary.add(word)

    def extract_features(self, sentence: list[str]) -> Counter:
        features = Counter()
        for word in sentence:
          if word in self.vocabulary:
            features[word]+=1
        return features

class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, max_num_features: int = 500):
        self.max_num_features = max_num_features
        self.vocabulary = set([])

    def build_vocabulary(self, examples: list[Example]):
        cnt = Counter()
        for example in examples:
          words = example.get_combined_words()
          for i in range(len(words)-1):
            bigram = words[i]+ " "+words[i+1]
            cnt[bigram]+=1

        for bigram, _ in cnt.most_common(self.max_num_features):
          self.vocabulary.add(bigram)

    def extract_features(self, sentence: list[str]) -> Counter:
        features = Counter()
        for i in range(len(sentence)-1):
          bigram = sentence[i]+" "+sentence[i+1]
          if bigram in self.vocabulary:
            features[bigram]+=1
        return features


class FancyFeatureExtractor(FeatureExtractor):
    def __init__(self, max_num_features: int = 500):
        self.max_num_features = max_num_features
        self.vocabulary = set([])

    def build_vocabulary(self, examples: list[Example]):
        negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                          "nowhere", "nor", "cannot", "can't", "won't", "don't",
                          "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
        cnt = Counter()
        for example in examples:
            words = example.get_combined_words()

            # Feature 1: Negation-prefixed unigrams
            negate = False
            for word in words:
                if word in negation_words:
                    negate = True
                elif negate:
                    cnt["NOT_" + word] += 1
                    negate = False

            # Feature 2: Overlap tokens (OVERLAP_word)
            mid = len(words) // 2
            premise_set = set(words[:mid])
            hypothesis_set = set(words[mid:])
            for w in premise_set & hypothesis_set:
                cnt["OVERLAP_" + w] += 1

            # Feature 3: Hypothesis-only words (HYP_ONLY_word)
            for w in hypothesis_set - premise_set:
                cnt["HYP_ONLY_" + w] += 1

            # Feature 4: Premise-only words (PREM_ONLY_word)
            for w in premise_set - hypothesis_set:
                cnt["PREM_ONLY_" + w] += 1

            # Feature 5: Trigrams
            for i in range(len(words) - 2):
                trigram = words[i] + " " + words[i+1] + " " + words[i+2]
                cnt[trigram] += 1

        for feat, _ in cnt.most_common(self.max_num_features):
            self.vocabulary.add(feat)

    def extract_features(self, sentence: list[str]) -> Counter:
        negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                          "nowhere", "nor", "cannot", "can't", "won't", "don't",
                          "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"}
        features = Counter()

        mid = len(sentence) // 2
        premise_set = set(sentence[:mid])
        hypothesis_set = set(sentence[mid:])

        # Feature 1: Negation-prefixed unigrams
        negate = False
        for word in sentence:
            if word in negation_words:
                negate = True
            elif negate:
                feat = "NOT_" + word
                if feat in self.vocabulary:
                    features[feat] += 1
                negate = False

        # Feature 2: Overlap tokens
        for w in premise_set & hypothesis_set:
            feat = "OVERLAP_" + w
            if feat in self.vocabulary:
                features[feat] += 1

        # Feature 3: Hypothesis-only words
        for w in hypothesis_set - premise_set:
            feat = "HYP_ONLY_" + w
            if feat in self.vocabulary:
                features[feat] += 1

        # Feature 4: Premise-only words
        for w in premise_set - hypothesis_set:
            feat = "PREM_ONLY_" + w
            if feat in self.vocabulary:
                features[feat] += 1

        # Feature 5: Trigrams
        for i in range(len(sentence) - 2):
            trigram = sentence[i] + " " + sentence[i+1] + " " + sentence[i+2]
            if trigram in self.vocabulary:
                features[trigram] += 1

        # Scalar features (always included, not vocabulary-gated)
        total = len(premise_set | hypothesis_set)
        overlap_count = len(premise_set & hypothesis_set)
        features["__overlap_ratio__"] = overlap_count / total if total > 0 else 0.0
        features["__len_diff__"] = abs(len(sentence[:mid]) - len(sentence[mid:])) / (len(sentence) + 1)
        features["__has_negation__"] = 1 if any(w in negation_words for w in sentence) else 0

        return features
