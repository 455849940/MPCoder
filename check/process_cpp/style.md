# JAVA codeStyle check
## Structure 
noLineWrap |Never break import and package lines.|NoLineWrapCheck|
noStarImport| No .* imports.|AvoidStarImportCheck|
oneTopClass |Put a top class in its own file.|OneTopLevelClassCheck|
EmptyLineSeparator |Use a blank line after header, package, import lines, class, methods,fields, static, and instance initializers.|EmptyLineSeparatorCheck|
-------------------------------------------------------------------------------------------------------------------------------------------------------------
## Formatting 
WhitespaceAround  |Use a space between a reserved word and its follow-up bracket,e.g., if（.|WhitespaceAroundCheck
GenericWhitespace |Use a space before the definition of generic type, e.g., List <.|GenericWhitespaceCheck
OperatorWrap      |Break after ‘=’ but before other binary operators.|OperatorWrapCheck
SeparatorWrap     |Break after ‘,’ but before ‘.’.|SeparatorWrapDot,SeparatorWrapComma!!!!!
LineLength        |100 characters in maximum.|LineLengthCheck
LeftCurly         |Put ‘{’ on the same line of code.|LeftCurlyCheck
RightCurly        |Put ‘}’ on its own line.|RightCurlySame!!!!!!
EmptyBlock        |No empty block for control statements.|EmptyBlockCheck
NeedBraces        |Use ‘{}’ for single control statements.|NeedBracesCheck
indentation       |Set a basic offset as four spaces.|IndentationCheck
MultipleVariableDeclarations        |Put a variable declaration on its own line.|MultipleVariableDeclarationsCheck
OneStatementPerLine    |Each line holds one statement.|OneStatementPerLineCheck
upperEll          |Use ‘L’ for Long integer literals.|UpperEllCheck
ModifierOrder     |Follow the order: public, protected, private, abstract, static,final, transient, volatile, synchronized, native, strictfp.|ModifierOrderCheck
FallThrough       |Put a fall-through comment in a switch If a ‘case’ has no break,return, throw, or continue.|FallThroughCheck
MissingSwitchDefault    |not Use “default” in switch.|MissingSwitchDefaultCheck
<!--noTab  -->    |No ‘\t’ in source code.
<!--AnnotationLocation  -->     |Put each annotation one line before a class or a method.
-------------------------------------------------------------------------------------------------------------------------------------------------------------
## Naming 
TypeName |Be in UpperCamelCase, e.g., BinarySearchTree.|TypeNameCheck
MethodName |Be in lowerCamelCase, e.g., getName.|MethodNameCheck
MemberName |Be in lowerCamelCase, e.g., localAddress.|MemberNameCheck
ParameterName |Be in lowerCamelCase, e.g., customerId.|ParameterNameCheck
LocalVariableName |Be in lowerCamelCase, e.g., clientAccount.|LocalVariableNameCheck
<!--packageName  -->|Be in all lowercase, with consecutive words concatenated together by ‘.’, e.g., com.edu.nameusage.|
<!--CatchParameterName  -->|Be in lowerCamelCase or in one-character lowercase,e.g., divideZeroException or e.|CatchParameterNameCheck
-------------------------------------------------------------------------------------------------------------------------------------------------------------

# CPPLINT stylecheck

## Structure
'build/namespaces'       | Do not use namespace using-directives.Use using-declarations instead.
'runtime/explicit'       | Check if external constructor markings conform to the specifications.
'runtime/references'     | Make a variable const when don't intend to modify its referenced value 
'runtime/int'            | Use "unsigned short" for ports, not "short" | Use int16/int64/etc, rather than the C type
'runtime/arrays'         | Do not use variable-length arrays.  Use an appropriately named ('k' followed by CamelCase) compile-time constant for the size
'runtime/string'         | Check whether static/global string variables/constants conform to the specifications.

-------------------------------------------------------------------------------------------------------------------------------------------------------------

## Formatting
'whitespace/tab'         | Use only spaces, and indent 2 spaces at the begin of the line
'whitespace/operators'   | Checks for horizontal spacing around operators
'whitespace/braces'      | Checks for horizontal spacing near braces
'whitespace/comma'       | Should always have a space after a comma
'whitespace/indent'      | Check that access keywords are indented +1 space.(such as protected etc ..)
'whitespace/comments'    | Should have a space between // and comment, two spaces between code and comments
'whitespace/end_of_line' | Checks for lines ending in whitespace
'whitespace/parens'      | Checks for horizontal spacing around parentheses
'whitespace/semicolon'   | Checks for horizontal spacing near semicolons.
'whitespace/ending_newline'        | Should have a newline char at the end of the file
'whitespace/blank_line'  | Checks for redundant blank line in code block
'readability/braces'     | Check the usage of if-else statements with curly braces
'whitespace/line_length' | Lines should be <= 80 characters long
'whitespace/newline'     | Check if the new lines in loops and branching statements conform to the specifications.








# PYLINT stylecheck
## (C) convention, for programming standard violation
## (R) refactor, for bad code smell






