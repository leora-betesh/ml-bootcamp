## -*- coding: utf-8 -*-
#"""
#Created on Thu Feb 28 13:20:58 2019
#
#@author: Leora Betesh
#"""
#
#Part 1 – Python basic
## 1.1. This is the print statement
print("Hello world")
## GO!
## 1.2. This is a variable
message = "Level Two"
print(message)
# Add a line below to print this variable
## GO!
## 1.3. The variable above is called a string
## You can use single or double quotes (but must close them)
## You can ask Python what type a variable is. Try uncommenting the next line:
print(type(message))
## GO!
## 1.4. Another type of variable is an integer (a whole number)
a = 123
b = 654
c = a + b
## Try printing the value of c below to see the answer
## GO!
print(c)
## 1.6. Variables keep their value until you change it
a = 100
print(a) # think - should this be 123 or 100?
c = 50
print(c) # think - should this be 50 or 777?
d = 10 + a - c
print(d) # think - what should this be now?
## GO!
## 1.7. You can also use '+' to add together two strings
greeting = 'Hi '
name = 'Leora' # enter your name in this string
message = greeting + name
print(message)
## GO!
## 1.8. Try adding a number and a string together and you get an error:
age = 39 # enter your age here (as a number)
#print(name + ' is ' + age + ' years old')
## GO!
## See the error? You can't mix types like that.
## But see how it tells you which line was the error?
## Now comment out that line so there is no error
## 1.9. We can convert numbers to strings like this:
print(name + ' is ' + str(age) + ' years old')
## GO!
## No error this time, I hope?
## Or we could just make sure we enter it as a string:
age = '39' # enter your age here, as a string
print(name + ' is ' + age + ' years old')
## GO!
## No error this time, I hope?
## 1.10. Another variable type is called a boolean
## This means either True or False
raspberry_pi_is_fun = True
raspberry_pi_is_expensive = False
## We can also compare two variables using ==
bobs_age = 15
your_age = 39 # fill in your age
print(your_age == bobs_age) # this prints either True or False
## GO!
## 1.11. We can use less than and greater than too - these are < and >
bob_is_older = bobs_age > your_age
print(bob_is_older) # do you expect True or False?
## GO!
## 1.12. We can ask questions before printing with an if statement
money = 500
phone_cost = 240
tablet_cost = 260
total_cost = phone_cost + tablet_cost
can_afford_both = money > total_cost
if can_afford_both:
    message = "You have enough money for both"
else:
    message = "You can't afford both devices"
print(message) # what do you expect to see here?
## GO!
## Now change the value of tablet_cost to 260 and run it again
## What should the message be this time?
## GO!
## Is this right? You might need to change the comparison operator to >=
## This means 'greater than or equal to'
raspberry_pi = 25
pies = 3 * raspberry_pi
total_cost = total_cost + pies
if total_cost <= money:
    message = "You have enough money for 3 raspberry pies as well"
else:
    message = "You can't afford 3 raspberry pies"
print(message) # what do you expect to see here?
## GO!
## 1.13. You can keep many items in a type of variable called a list
colours = ['Red', 'Orange', 'Yellow', 'Green', 'Blue', 'Indigo', 'Violet']
## You can check whether a colour is in the list
print('Black' in colours) # Prints True or False
## GO!
## You can add to the list with append
colours.append('Black')
colours.append('White')
print('Black' in colours) # Should this be different now?
## GO!
## You can add a list to a list with extend
more_colours = ['Gray', 'Navy', 'Pink']
colours.extend(more_colours)
## Try printing the list to see what's in it
print(colours)
## GO!
## 1.14. You can add two lists together in to a new list using +
primary_colours = ['Red', 'Blue', 'Yellow']
secondary_colours = ['Purple', 'Orange', 'Green']
main_colours = primary_colours + secondary_colours
## Try printing main_colours
print(main_colours)
## 1.15. You can find how many there are by using len(your_list). Try it below
## How many colours are there in main_colours?
print(len(main_colours))
## GO!
all_colours = colours + main_colours
## How many colours are there in all_colours?
print(len(all_colours))
## Do it here. Try to think what you expect before you run it
## GO!
## Did you get what you expected? If not, why not?
## 1.16. You can make sure you don't have duplicates by adding to a set
even_numbers = [2, 4, 6, 8, 10, 12]
multiples_of_three = [3, 6, 9, 12]
numbers = even_numbers + multiples_of_three
print(numbers, len(numbers))
numbers_set = set(numbers)
print(numbers_set, len(numbers_set))
## GO!
colour_set = set(all_colours)
## How many colours do you expect to be in this time?
## Do you expect the same or not? Think about it first
print(len(colour_set))
## 1.17. You can use a loop to look over all the items in a list
my_class = ['Sarah', 'Bob', 'Jim', 'Tom', 'Lucy', 'Sophie', 'Liz', 'Ed']
## Below is a multi-line comment
## Delete the ''' from before and after to uncomment the block
#'''
for student in my_class:
    print(student)
#'''
## Add all the names of people in your group to this list
my_class.extend(['Liat','Gil','Kobi','Yael','Michal','Leora'])
## Remember the difference between append and extend. You can use either.
## Now write a loop to print a number (starting from 1) before each name
for i in range(len(my_class)):
    print(i+1,my_class[i])
    
## 1.18. You can split up a string by index
full_name = 'Dominic Adrian Smith'
first_letter = full_name[0]
last_letter = full_name[19]
first_three = full_name[:3] # [0:3 also works]
last_three = full_name[-3:] # [17:] and [17:20] also work
middle = full_name[8:14]
print(first_letter,last_letter,first_three,last_three,middle)

new_word = full_name[12] + full_name[6] + full_name[6] + full_name[1] + full_name[10] + full_name[9] + full_name[11:13] + full_name[4]
print(new_word)
## Try printing these, and try to make a word out of the individual letters
## 1.19. You can also split the string on a specific character
my_sentence = "Hello, my name is Fred"
parts = my_sentence.split(',')
print(parts)
print(type(parts)) # What type is this variable? What can you do with it?
## GO!
my_long_sentence = "This is a very very very very very very long sentence"
## Now split the sentence and use this to print out the number of words
words = my_long_sentence.split(' ')
print(len(words))
## GO! (Clues below if you're stuck)
## Clue: Which character do you split on to separate words?
## Clue: What type is the split variable?
## Clue: What can you do to count these?
## 1.20. You can group data together in a tuple
person = ('Bobby', 26)
print(person[0] + ' is ' + str(person[1]) + ' years old')
## GO!
# (name, age)
students = [
 ('Dave', 12),
 ('Sophia', 13),
 ('Sam', 12),
 ('Kate', 11),
 ('Daniel', 10)
]
## Now write a loop to print each of the students' names and age
## GO!
for name, age in students:
    print(name,age)
## 1.21. Tuples can be any length. The above examples are 2-tuples.
## Try making a list of students with (name, age, favourite subject and sport)
students = [
 ('Dave', 12,'history','boxing'),
 ('Sophia', 13,'math','fencing'),
 ('Sam', 12,'science','soccer'),
 ('Kate', 11,'english','basketball'),
 ('Daniel', 10,'hebrew','volleyball')
]
## Now loop over them printing each one out
for name,age,subject,sport in students:
    print(name,age,subject,sport)
## Now pick a number (in the students' age range)
## Make the loop only print the students older than that number
for name,age,subject,sport in students:
    if age > 11:
        print(name,age,subject,sport)
## GO!
## 22. Another useful data structure is a dictionary
## Dictionaries contain key-value pairs like an address book maps name
## to number
addresses = {
 'Lauren': '0161 5673 890',
 'Amy': '0115 8901 165',
 'Daniel': '0114 2290 542',
 'Emergency': '999'
}
## You access dictionary elements by looking them up with the key:
print(addresses['Amy'])
### You can check if a key or value exists in a given dictionary:
print('David' in addresses) # [False]
print('Daniel' in addresses) # [True]
print('999' in addresses) # [False]
print('999' in addresses.values()) # [True]
print(999 in addresses.values()) # [False]
## GO!
## Note that 999 was entered in to the dictionary as a string, not an integer
## Think: what would happen if phone numbers were stored as integers?
## Try changing Amy's phone number to a new number
addresses['Amy'] = '0115 236 359'
print(addresses['Amy'])
## GO!
## Delete Daniel from the dictinary
print('Daniel' in addresses) # [True]
del addresses['Daniel']
print('Daniel' in addresses) # [False]
## GO!
## You can also loop over a dictionary and access its contents:
#'''
for name in addresses:
    print(name, addresses[name])
#'''
## GO!
## 1.23. A final challenge using the skills you've learned:
## What is the sum of all the digits in all the numbers from 1 to 1000?
## GO!
totalSum = 0
for i in range(1000):
    currNum = str(i)
    for char in currNum:
        totalSum += int(char)
        
print(totalSum)
## Clue: range(10) => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
## Clue: str(87) => '87'
## Clue: int('9') => 9
#1.24. Define a function `max()` that takes two numbers as arguments and
#returns the largest of them. Use the if-then-else construct available in Python.
#(It is true that Python has the `max()` function built in, but writing it yourself is
#nevertheless a good exercise ).
def max(x,y):
    if y>x:
        return y
    else:
        return x

print(max(21,22))
#1.25. Define a function `max_of_three()` that takes three numbers as
#arguments and returns the largest of them.
def max_of_three(x,y,z):
    if max(x,y) == max(x,z):
        return x
    else:
        return max(y,z)

print(max_of_three(23,21,22))

#1.26. Define a function that computes the length of a given list or string. ( It is
#true that Python has the `len()` function built in, but writing it yourself is
#nevertheless a good exercise ).

def list_len(lst):
    counter = 0
    for i in lst:
        counter +=1
    return counter

print(list_len("Leora Betesh")) 

#1.27. Write a function that takes a character ( i.e. a string of length 1 ) and
#returns `True` if it is a vowel, `False` otherwise.
def isVowel(char):
    if char in 'aeiou':
        return True
    return False

print(isVowel('i'))

def isConsonant(char):
    if char in 'bcdfghjklmnpqrstvwxyz':
        return True
    return False


#1.28. Write a function `translate()` that will translate a text into "rövarspråket"
#(Swedish for "robber's language"). That is, double every consonant and place
#an occurrence of "o" in between. For example, `translate("this is fun")` should
#return the string `"tothohisos isos fofunon".`

def translate(text):
    textList = list(text)
    newText = []
    for i in range(len(textList)):
        if isConsonant(textList[i]):
            newText.append(textList[i])
            newText.append('o')
            newText.append(textList[i])
        else:
            newText.append(textList[i])
    return ''.join(newText)

print(translate("this is fun"))

#Part 2 - advanced
#2.1
#Write a program which will find all such numbers which are divisible by 7 but
#are not a multiple of 5,
#between 2000 and 3200 (both included).
#The numbers obtained should be printed in a comma-separated sequence on
#a single line.
#Hints:
#Consider use range(#begin, #end) method
def weird_nums():
    for i in range(2000,3200):
        if i % 7 == 0 and i % 5 != 0:
            if i == 3199:
                print(i, end=' ')
            else:
                print(i, end=', ', flush=True)
    
weird_nums()
#2.2
#Write a program which can compute the factorial of a given numbers.
#The results should be printed in a comma-separated sequence on a single
#line.
#Suppose the following input is supplied to the program:
#8
#Then, the output should be:
#40320
#Hints:
#In case of input data being supplied to the question, it should be assumed to
#be a console input.
def factorial(n):
    fact = 1
    for i in range(1,n+1):
        fact = fact * i
        print(fact, end=', ', flush=True)
        
factorial(8)
#2.3
#With a given integral number n, write a program to generate a dictionary that
#contains (i, i*i) such that is an integral number between 1 and n (both
#included). and then the program should print the dictionary.
#Suppose the following input is supplied to the program:
#8
#Then, the output should be:
#{1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49, 8: 64}
#Hints:
#In case of input data being supplied to the question, it should be assumed to
#be a console input.
#Consider use dict()

def integral_dict(n):
    
    integral_list = {}
    for i in range(1,n+1):
        integral_list[i] = i*i
    return integral_list


print(integral_dict(8))

#2.4
#Write a program which accepts a sequence of comma-separated numbers
#from console and generate a list and a tuple which contains every number.
#Suppose the following input is supplied to the program:
#34,67,55,33,12,98
#Then, the output should be:
#['34', '67', '55', '33', '12', '98']
#('34', '67', '55', '33', '12', '98')
#Hints:
#In case of input data being supplied to the question, it should be assumed to
#be a console input.
#tuple() method can convert list to tuple

n = input().split(',')
print(n)
print(tuple(n))
#2.5
#Write a method which can calculate a square value of a number
def calc_square(n):
    """ Retun the square of the input number """
    return n*n

print(calc_square(-2.5))
#2.6
#Python has many built-in functions, and if you do not know how to use it, you
#can read document online or find some books. But Python has a built-in
#document function for every built-in functions.
#Please write a program to print some Python built-in functions documents, such
#as abs(), int(), raw_input()
#And add document for your own function
#
#Hints:
# The built-in document method is __doc__

def get_doc(func):
    print(func.__doc__)

get_doc(calc_square)

#2.7.
#class
#Answer the follwing questions and write the following classes
#1. What is a class?
#2. What is an instance?
#3. What is encapsulation?
#4. What is inheritance?
#5. What is polymorphism?
#6. Traingle Class:
#a. Create a class, Triangle. Its __init__() method should take self,
#angle1, angle2, and angle3 as arguments. Make sure to set these
#appropriately in the body of the __init__()method.

class Triangle:
    def __init__(self, angle1, angle2, angle3):
        self.angle1 = angle1
        self.angle2 = angle2
        self.angle3 = angle3
    
    def check_angles(self):
        if self.angle1 + self.angle2 + self.angle3 == 180:
            return True
        return False
    
    num_of_sides = 3
    
t1 = Triangle(90, 30, 70)
print(t1.check_angles())
#b. Create a variable named number_of_sides and set it equal to 3.

#c. Create a method named check_angles. The sum of a triangle's
#three angles is It should return True if the sum of self.angle1,
#self.angle2, and self.angle3 is equal 180, and False otherwise.
#d. Create a variable named my_triangle and set it equal to a new
#instance of your Triangle class. Pass it three angles that sum to 180
#(e.g. 90, 30, 60).

#e. Print out my_triangle.number_of_sides and print out
#my_triangle.check_angles().
print(t1.num_of_sides)
print(t1.check_angles())
#7. Songs class:
#a. Define a class called Songs, it will show the lyrics of a song. Its
#__init__() method should have two arguments:selfanf lyrics. Lyrics is
#a list. Inside your class create a method called sing_me_a_song that
#prints each element of lyrics on his own line. Define a varible:
#b. happy_bday = Song(["May god bless you, ",
# "Have a sunshine on you,",
# "Happy Birthday to you !"])
#c. Call the sing_me_song method on this variable
class Songs:
    def __init__(self, lyrics):
        self.lyrics = lyrics

    def sing_me_a_song(self):
        for stanza in self.lyrics:
            print(stanza)
                
happy_bday = ["May god bless you, ",
                   "Have a sunshine on you,",
                   "Happy Birthday to you !"]

abc = ["ABCDEFG","HIJKLMNOP","QRSTUV","WXY and Z","Now I know my abc's.","Next time won't you sing with me?"]
Song1 = Songs(happy_bday)
Song2 = Songs(abc)
Song1.sing_me_a_song()
Song2.sing_me_a_song()

#8. Define a class called Lunch. Its __init__() method should have two
#arguments:selfanf menu.Where menu is a string. Add a method called
#menu_price.It will involve a ifstatement:
#if "menu 1" print "Your choice:", menu, "Price 12.00", if "menu 2" print
#"Your choice:", menu, "Price 13.40", else print "Error in menu". To check
#if it works define: Paul=Lunch("menu 1") and call Paul.menu_price().

class Lunch:
    def __init__(self, menu):
       self.menu = menu

    def menu_price(self):
        if self.menu == "menu 1":
            print("Your choice:", self.menu, "Price 12.00")
        elif self.menu == "menu 2":
            print("Your choice:", self.menu, "Price 13.40")
        else:
            print("Error in menu")
            
Paul=Lunch("menu 1")
Paul.menu_price()

#9. Define a Point3D class that inherits from object Inside the Point3D class,
#define an __init__() function that accepts self, x, y, and z, and assigns
#these numbers to the member variables self.x,self.y,self.z. Define a
#__repr__() method that returns "(%d, %d, %d)" % (self.x, self.y, self.z).
#This tells Python to represent this object in the following format: (x, y, z).
#Outside the class definition, create a variable named my_point containing
#a new instance of Point3D with x=1, y=2, and z=3. Finally, print
#my_point.
#You can find the solutions to these exercises in the link:
#https://erlerobotics.gitbooks.io/erle-robotics-learning-python-gitbookfree/classes/exercisesclasses.html
class Point3D(object):
    def __init__(self, x,y,z):
       self.x = x
       self.y = y
       self.z = z
       
    def __repr__(self):
        return("{}('{}')".format(self.__class__.__name__, self.x, self.y, self.z))
    
my_point = Point3D(1,2,3)
my_point

#2.8.
#List comprehension - a quick reminder
#1. Use dictionary comprehension to create a dictionary of numbers 1- 100 to
#their squares (i.e. {1:1, 2:4, 3:9 …}


def square_dict(n):
    square_list = {i: i*i for i in range(1,101)}
    return square_list

print(square_dict(100))

#2. Use list comprehension to create a list of prime numbers in the range 1-
#100
#http://www.secnetix.de/olli/Python/list_comprehensions.hawk
import time
def prime_num(n):
    prime_num = [i for i in range(2,n+1) if (i==2 or i==3 or i==5 or i==7) or (i%2 and i%3 and i%5 and i%7) ]       
    return prime_num

def prime_num2(n):
    prime_num = []
    for i in range(2,n):
        isPrime = True
        for j in range(2,i-1):
            if i%j ==0:
                isPrime = False
        if isPrime == True:
            prime_num.append(i)
    return prime_num
        
start_time1 = time.time()
method1 = prime_num(10000)
#print(prime_num(10000))
print("time elapsed: {:.2f}s".format(time.time() - start_time1))

start_time2 = time.time()
method2 = prime_num2(10000)
#print(prime_num2(10000))
print("time elapsed 2: {:.2f}s".format(time.time() - start_time2))


#2.9.
#Files
#Use open, read, write, close which are explained in these tutorials:
#https://www.tutorialspoint.com/python/python_files_io.htm
#https://www.guru99.com/reading-and-writing-files-in-python.html
#to:
#1. Write “Hello world” to a file
#2. Read the file back into python.
#3. Write the square roots of the numbers 1-100 into the file, each in new
#line.

import math

myfile = open("fileIO.txt","w")
myfile.write("Hello World")
myfile.close()

myfile = open("fileIO.txt","a+")
for i in range (1,101):
    myfile.write(str(math.sqrt(i)))
    myfile.write("\n")
myfile.close()

filetext = open("fileIO.txt", "r") 
print(filetext.read())
filetext.close()

#2.10
#With
#Read about with in one of the following links:
#https://stackoverflow.com/questions/1369526/what-is-the-python-keywordwith-used-for
#http://effbot.org/zone/python-with-statement.htm
#1. Open a file using with and write the first paragraph from (you don’t need
#to read the webpage, just copy paste it into code)
#https://en.wikipedia.org/wiki/Eigenface
eigen_text = "Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition. The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification. The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set."
with open("eigen.txt", "w") as f:
    f.write(eigen_text)
f.close()

#2. Open the file using with and read the contents using readlines.
with open("eigen.txt", "r") as ef:
    for line in ef:
        print(line)
ef.close()
#2.11.
#Strings
#1. Split the paragraph to a list of words by spaces (hint: split())
eigen_text = "Eigenfaces is the name given to a set of eigenvectors when they are used in the computer vision problem of human face recognition.[1] The approach of using eigenfaces for recognition was developed by Sirovich and Kirby (1987) and used by Matthew Turk and Alex Pentland in face classification.[2] The eigenvectors are derived from the covariance matrix of the probability distribution over the high-dimensional vector space of face images. The eigenfaces themselves form a basis set of all images used to construct the covariance matrix. This produces dimension reduction by allowing the smaller set of basis images to represent the original training images. Classification can be achieved by comparing how faces are represented by the basis set."
eigen_sentences = eigen_text.split(".")
reversed_sentences = []
for sentence in eigen_sentences:
    eigen_words = sentence.split()
    eigen_words.reverse()
    rev_sentence = ' '.join(eigen_words)
    reversed_sentences.append(rev_sentence)
eigen_para = '. '.join(reversed_sentences)
print(eigen_para)

#2. Join the words back into a long string using “join”. (hint: join())
#* (optional) Create a paragraph with the word order reversed in each
#sentence (but keep the order of the sentences)

#3. write a function accepting two numbers and printing “the sum of __ and
#___ is ___”.
#Do this twice using one string formats learned in class each time.
#Hint: https://pyformat.info/
def string_sum(a,b):
    print("The sum of {0} and {1} is {2}".format(a,b,a+b))
   # print("The sum of {0} and {1} is {sum(a,b)}".format(a,b))
string_sum(4,5)


#2.12.
#Datetime, time, timeit
#Read the current datetime to a variable usint datetime.datetime.no see:
#https://docs.python.org/3/library/datetime.html
#10. Print the date of the day before yesterday, and the datetime in 12 hours
#from now
#Use time library to time library to time operations by storing time.time()
#before and after running an operation
#https://www.tutorialspoint.com/python/time_time.htm
#1. Time one of the comprehension exercises you’ve performed and compare
#to a simple for implementation, to determine which one is faster
#1. Now use timeit library to time your list comprehension operation, see:
#https://docs.python.org/2/library/timeit.html
import datetime
import timeit
now = datetime.datetime.now()
date_two_days_ago = datetime.datetime.now() - datetime.timedelta(days=2)
twelve_hours_from_now = datetime.datetime.now() + datetime.timedelta(hours=12)
print("now",now)
print("date_two_days_ago:",date_two_days_ago.date())
print("12 hours from now:",twelve_hours_from_now.time())

timeit.Timer('for i in range(10): oct(i)', 'gc.enable()').timeit()

#2.13.
#Exceptions
#Read about exceptions from:
#http://www.pythonforbeginners.com/error-handling/exception-handling-inpython
#1. Enhance the function you wrote which reads numbers (Strings 3.)
#a. In case of a string input writing an error message instead of raising
#an error
#b. Raise an error in case one of the input numbers is > 100

def string_sum(a,b):
    if isinstance(a,str) or isinstance(b,str) :
        print("The inputs must be numbers.  You supplied a string.")
    elif a > 100:
        raise Exception('The input should not exceed 100. The value entered was: {}'.format(a))
    elif b > 100:
        raise Exception('The input should not exceed 100. The value entered was: {}'.format(b))
    else:
        print("The sum of {0} and {1} is {sum(a,b)}".format(a,b))
    

string_sum(4,"1")

#c. Create a code which tries to read from the dictionary of squares created
square_list = {i: i*i for i in range(1,101)}

try:
    item = square_list[200]
except KeyError:
    print("Dictionary does not contain that value")

#2.14.
#Logger
#Read about loggers from the example:
#https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
#Create a logger in any of your programs and log various messages to a file
#using a FileHandler, and to a the console using StreamHandler
#*Optional : try setting different levels (e.g. debug for console and info for file)
#and different formats and try different levels of logs in your code.
import logging
        
logger = logging.getLogger(__name__)

if logger.handlers:
    logger.handlers = []
    
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('string_sum.txt')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.CRITICAL)

logger.addHandler(file_handler)    
logger.addHandler(stream_handler)

# This is the class whose objects will be used in the application code directly to call the functions.
def string_sum(a,b):
    
    logger.debug('Entered string_sum function')
    
    if isinstance(a,str) or isinstance(b,str) :
        logger.critical("The inputs must be numbers.  You supplied a string.")
    elif a > 100 or b > 100:
        logger.error("The value may not exceed 100")
    else:
        logger.info("Sum was calculated")
        c = a+b
        print("The sum of {0} and {1} is {2}".format(a,b,c))
    
string_sum(4,1)
string_sum(4,1200)
string_sum(4,"1")


#2.15
#Regular expressions (re)
#You can read about regex from the following sources:
#Python documentation - https://docs.python.org/3/library/re.html
#General links about RegEx - http://www.regular-expressions.info/tutorial.html
#And - https://www.icewarp.com/support/online_help/203030104.htm
#(The last is more brief and practical)
#Try to understand the difference between search and match (and compile),
#and how to match the strings exactly.
#1. To get the hang of it, try to do a few of the the exercises in:
#https://regexone.com/
#2. To get a sense of regex in python, take a look at the first few exercises in:
#http://www.w3resource.com/python-exercises/re/

#1. Write a Python program to check that a string contains only a certain set of characters (in this case a-z, A-Z and 0-9).

def verify_chars(string):
    import re
    pattern = r'[^\.a-zA-Z0-9]'
    if re.search(pattern, string):
        #Character other then . a-z 0-9 was found
        return "Invalid"
    else:
        #No character other then . a-z 0-9 was found
        return "valid"

print(verify_chars("LB%1"))
# 3*. Optional: Write a regular expression for identifying email addresses,
#find and compare to regular expression found online.

def verify_email(address):
    import re
    pattern = r'^\w+@[a-zA-Z_]+?\.[a-zA-Z]{2,3}$'
    if re.search(pattern, address):
        return "Valid"
    else:
        return "Invalid"

print(verify_email("lb@gmail.com"))

#2.16.
#Combine everything!
#Write a simple “range calculator” function which reads a file and writes the
#result into another file. You should log the operations, and how much time
#they took and handle problematic inputs (strings, wrong characters, etc.).
#Validate the input using regex.
#The file will contain in each line two numbers denoting a range of numbers
#and they can only be integers, an operation denoted by a sign (+,*,-,/,**) and
#a third number which will be applied with the operation for each of the
#numbers in the range. Note all numbers and signs are separated by blanks
#(please don’t use .csv readers and use plain readers):
#5 100 - 17
#18 25 * 2.784
#(First line: subtract 17 from all numbers between 5 and 100, Second line:
#multiply by 2.784 all numbers between 18 and 25)
#The result of each range should be written into one line, separated with
#commas and with the precision of two digits after the decimal point.
import time
import re
import operator
import logging
import os
        
logger = logging.getLogger(__name__)

if logger.handlers:
    logger.handlers = []
    
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')

file_handler = logging.FileHandler('string_sum.txt')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.ERROR)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
stream_handler.setLevel(logging.CRITICAL)

logger.addHandler(file_handler)    
logger.addHandler(stream_handler)

ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow
}   
#os.chdir("C:/LeoraProjects/bootcamp/")
print(os.getcwd())
def range_calculator(input_file,output_file):
    patternNum = r'[^\.0-9]'
    patternOp = r'[^\.*+-/]'
    operators = []
    fo = open(output_file,"a+")
    with open(input_file, "r") as f:
        for line in f:
            operators = line.split()
            if not(re.search(patternNum, operators[0])) or not(re.search(patternNum, operators[1])):
                logger.critical("You must enter integers as operands")
                f.close()
                return -1
            if not(re.search(patternOp, operators[2])):
                logger.critical("Operator must be * + - \ or **")
                f.close()
                return -1
            op1 = int(operators[0])
            op2 = int(operators[1])
            op4 = float(operators[3])
            op_func = ops[operators[2]]
            for x in range(op1,op2+1):
                result = op_func(x,op4)                
                fo.write(str(round(result,2)))
                if x < op2:
                    fo.write(",")
            fo.write("\n")
    f.close()
    fo.close()
    
    
            
start_time = time.time()
range_calculator("in.txt","out.txt")
end_time = time.time()
print(start_time - end_time)
#2.17.
#os
#Os module provides a portable way of using operating system dependent
#functionality you can read about os in this link.
#https://docs.python.org/3/library/os.html
#1. print absolute path on your system
#2. print files and directories in the current directory on your system 

import os 

print("path of this file",os.path.dirname(os.path.realpath(__file__)))

print("current working directory:",os.getcwd())

print(os.listdir(os.getcwd()) )

#2.18.
#glob.
#The glob module finds all the pathnames matching a specified pattern
#according to the rules used by the Unix shell.
#You can read about it in the next link.
#https://docs.python.org/3/library/glob.html
#Find in your download directory all the files that have extension pdf (.pdf)
import os
import glob

os.chdir("C:/Users/User1/Downloads")
for file in glob.glob("*.pdf"):
    print(file)

#2.19.
#enumerate
#http://book.pythontips.com/en/latest/enumerate.html
#Write a list with some values, use for and enumerate to print in each iteration
#the value and his index

nums = (4,3,6,8,9)
for counter, value in enumerate(nums):
    print(counter, value)
#2.20.
#Threads
#1. Write two functions, one that writes “SVD” article from wikipedia to a file
#and one that calculates the sum of two numbers. Run the two functions
#simultaneously using threads.
#2. Use Threading.Lock() in order to acquire and release mutexes so the two
#functions don’t run simultaneously.
#Hint: http://www.python-course.eu/threads.php

import urllib
import threading

mutex = threading.Lock()

def write_to_file(url,outfile,mutex):
    mutex.acquire()
    response = urllib.request.urlopen(url)
    html = response.read()
    myfile = open(outfile,"a+")
    myfile.write(str(html))
    print("finished writing html")
    myfile.close()
    response.close() 
    mutex.release()
    
def calc_sum(a,b,mutex):
    mutex.acquire()
    print(a+b)
    mutex.release()
    
thread1 = threading.Thread(target = write_to_file, args=("http://python.org/","svd.txt",mutex))
thread2 = threading.Thread(target = calc_sum, args=(100000000,200000,mutex))
thread1.start()
thread2.start()
print("threading example finished")

#2.21
#Random
#https://docs.python.org/3/library/random.html
#Create a list with 10 values (string)
#Take only one value from the list from a random place
#Split the list to two lists in different sizes each list contain values from the
#previous list but in random order.
import random
import numpy as np
list_orig = ["One", "Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"]
index1 = random.randint(0,9)
index2 = random.randint(0,9)
list1 = list_orig[0:index1]
list2 = list_orig[index2:]
np.random.shuffle(list1)
np.random.shuffle(list2)
print(list1)
print(list2)
