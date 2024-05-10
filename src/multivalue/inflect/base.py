#!/usr/bin/env python

# Copyright (c) 2006 Bermi Ferrer Martinez
# bermi a-t bermilabs - com
# See the end of this file for the free software, open source license
# (BSD-style).

import re


class Base:
    """Locale inflectors must inherit from this base class inorder to provide
    the basic Inflector functionality"""

    def conditionalPlural(self, numer_of_records, word):
        """Returns the plural form of a word if first parameter is greater than 1"""

        if numer_of_records > 1:
            return self.pluralize(word)
        else:
            return word

    def titleize(self, word, uppercase=""):
        """Converts an underscored or CamelCase word into a English sentence.
        The titleize function converts text like "WelcomePage",
        "welcome_page" or  "welcome page" to this "Welcome Page".
        If second parameter is set to 'first' it will only
        capitalize the first character of the title."""

        if uppercase == "first":
            return self.humanize(self.underscore(word)).capitalize()
        else:
            return self.humanize(self.underscore(word)).title()

    def camelize(self, word):
        '''Returns given word as CamelCased
        Converts a word like "send_email" to "SendEmail". It
        will remove non alphanumeric character from the word, so
        "who's online" will be converted to "WhoSOnline"'''
        return "".join(w[0].upper() + w[1:] for w in re.sub("[^A-Z^a-z^0-9^:]+", " ", word).split(" "))

    def underscore(self, word):
        """Converts a word "into_it_s_underscored_version"
        Convert any "CamelCased" or "ordinary Word" into an
        "underscored_word".
        This can be really useful for creating friendly URLs."""

        return re.sub(
            "[^A-Z^a-z^0-9^\/]+",
            "_",
            re.sub("([a-z\d])([A-Z])", "\\1_\\2", re.sub("([A-Z]+)([A-Z][a-z])", "\\1_\\2", re.sub("::", "/", word))),
        ).lower()

    def humanize(self, word, uppercase=""):
        """Returns a human-readable string from word
        Returns a human-readable string from word, by replacing
        underscores with a space, and by upper-casing the initial
        character by default.
        If you need to uppercase all the words you just have to
        pass 'all' as a second parameter."""

        if uppercase == "first":
            return re.sub("_id$", "", word).replace("_", " ").capitalize()
        else:
            return re.sub("_id$", "", word).replace("_", " ").title()

    def variablize(self, word):
        '''Same as camelize but first char is lowercased
        Converts a word like "send_email" to "sendEmail". It
        will remove non alphanumeric character from the word, so
        "who's online" will be converted to "whoSOnline"'''
        word = self.camelize(word)
        return word[0].lower() + word[1:]

    def tableize(self, class_name):
        """Converts a class name to its table name according to rails
        naming conventions. Example. Converts "Person" to "people" """
        return self.pluralize(self.underscore(class_name))

    def classify(self, table_name):
        """Converts a table name to its class name according to rails
        naming conventions. Example: Converts "people" to "Person" """
        return self.camelize(self.singularize(table_name))

    def ordinalize(self, number):
        """Converts number to its ordinal English form.
        This method converts 13 to 13th, 2 to 2nd ..."""
        tail = "th"
        if number % 100 == 11 or number % 100 == 12 or number % 100 == 13:
            tail = "th"
        elif number % 10 == 1:
            tail = "st"
        elif number % 10 == 2:
            tail = "nd"
        elif number % 10 == 3:
            tail = "rd"

        return str(number) + tail

    def unaccent(self, text):
        """Transforms a string to its unaccented version.
        This might be useful for generating "friendly" URLs"""
        find = "\u00c0\u00c1\u00c2\u00c3\u00c4\u00c5\u00c6\u00c7\u00c8\u00c9\u00ca\u00cb\u00cc\u00cd\u00ce\u00cf\u00d0\u00d1\u00d2\u00d3\u00d4\u00d5\u00d6\u00d8\u00d9\u00da\u00db\u00dc\u00dd\u00de\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e5\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f0\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f8\u00f9\u00fa\u00fb\u00fc\u00fd\u00fe\u00ff"
        replace = "AAAAAAACEEEEIIIIDNOOOOOOUUUUYTsaaaaaaaceeeeiiiienoooooouuuuyty"
        return self.string_replace(text, find, replace)

    def string_replace(self, word, find, replace):
        """This function returns a copy of word, translating
        all occurrences of each character in find to the
        corresponding character in replace"""
        for k in range(0, len(find)):
            word = re.sub(find[k], replace[k], word)

        return word

    def urlize(self, text):
        """Transform a string its unaccented and underscored
        version ready to be inserted in friendly URLs"""
        return re.sub("^_|_$", "", self.underscore(self.unaccent(text)))

    def demodulize(self, module_name):
        return self.humanize(self.underscore(re.sub("^.*::", "", module_name)))

    def modulize(self, module_description):
        return self.camelize(self.singularize(module_description))

    def foreignKey(self, class_name, separate_class_name_and_id_with_underscore=1):
        """Returns class_name in underscored form, with "_id" tacked on at the end.
        This is for use in dealing with the database."""
        if separate_class_name_and_id_with_underscore:
            tail = "_id"
        else:
            tail = "id"
        return self.underscore(self.demodulize(class_name)) + tail


# Copyright (c) 2006 Bermi Ferrer Martinez
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software to deal in this software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of this software, and to permit
# persons to whom this software is furnished to do so, subject to the following
# condition:
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THIS SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THIS SOFTWARE.
