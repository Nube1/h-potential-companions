%% =========================================================
%% Wits Thesis Class v6.9 - Full Math Thesis Ready (Fixed Roman Numerals)
%% (Modified to copy STB-Thesis style/font with Faint Gray Header/Right Footer)
%% =========================================================
\NeedsTeXFormat{LaTeX2e}[2020/08/10]
\ProvidesClass{witsthesis_stbcopy}[2025/11/04 v6.9 Full Math Thesis Fixed (STB Style/Font Copy)]

\LoadClass[12pt,oneside,openany]{book} % Added oneside, openany from stb-thesis

% =========================================================
% Geometry, fonts, spacing (STB Style Adjustments)
% =========================================================
\RequirePackage[a4paper,margin=1in,includeheadfoot]{geometry}
\RequirePackage{setspace}
\RequirePackage{microtype}
% Font Setup: Remove lmodern to use default Computer Modern (STB default)
% \RequirePackage{lmodern} % REMOVED
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\raggedbottom
\frenchspacing
% Spacing: Removed setstretch and parskip to match STB implied defaults
% \setstretch{1.5} % REMOVED
\setlength{\parindent}{1.5em}
% \setlength{\parskip}{0.5em} % REMOVED


% =========================================================
% Header/footer (STB Style ps@STBthesis and ps@plain + User Request)
% =========================================================
\RequirePackage{fancyhdr}
\RequirePackage{xcolor} % Required for light gray color
\definecolor{lightgray}{gray}{0.7} % Define a light gray color

% STB-Thesis Section Font Definitions (used for section headings only)
\newcommand{\STB@headfamily}{\normalfont\rmfamily}

% Redefine plain style to match the image: page number at bottom right.
\fancypagestyle{plain}{%
  \fancyhf{}%
  \renewcommand{\headrulewidth}{0pt}%
  \renewcommand{\footrulewidth}{0pt}%
  \fancyfoot[R]{\thepage}% % Page number at the bottom right
}

% Define the running head style to match the image: centered, light gray, underlined header; page number at bottom right.
\fancypagestyle{headings}{%
    \fancyhf{}%
    \renewcommand{\footrulewidth}{0pt}% % No rule on footer
    \renewcommand{\headrulewidth}{0.4pt}% % Small rule width for 'underlined'
    \colorlet{headrule}{lightgray} % Set rule color to light gray
    
    % Header: Centered, light gray, running title (rightmark)
    \fancyhead[C]{\color{lightgray}\MakeUppercase{\rightmark}}% 
    \fancyhead[R]{}% % Empty right header
    \fancyhead[L]{}% % Empty left header
    
    % Footer: Page number at the bottom right, normal font
    \fancyfoot[R]{\thepage}%
}
\pagestyle{headings} % Set default to the new running head style


% =========================================================
% Packages
% =========================================================
\RequirePackage{tikz}
\RequirePackage{graphicx}
\RequirePackage{textcase}
\RequirePackage{tcolorbox}
\RequirePackage{etoolbox}
\RequirePackage{amsthm}
\RequirePackage{thmtools}
\usetikzlibrary{calc}

% =========================================================
% Section Headings (STB Style)
% =========================================================
% Redefine section headings to use STB font, size, and spacing
\renewcommand\section{%
   \@startsection{section}{1}%
                 {\z@}%
                 {-3.5ex plus -1ex minus -.2ex}%
                 {2.3ex plus .2ex}%
                 {\STB@headfamily\Large\bfseries\raggedright}}
\renewcommand\subsection{%
   \@startsection{subsection}{2}%
                 {\z@}%
                 {-3.25ex plus -1ex minus -.2ex}%
                 {1.5ex plus .2ex}%
                 {\STB@headfamily\large\bfseries\raggedright}}%
\renewcommand\subsubsection{%
   \@startsection{subsubsection}{3}%
                 {\z@}%
                 {-3.25ex plus -1ex minus -.2ex}%
                 {1.5ex plus .2ex}%
                 {\STB@headfamily\normalsize\bfseries\raggedright}}
\renewcommand\paragraph{%
   \@startsection{paragraph}{4}%
                 {\z@}%
                 {3.25ex plus 1ex minus .2ex}%
                 {-1em}%
                 {\STB@headfamily\normalsize\bfseries}}
\renewcommand\subparagraph{%
   \@startsection{subparagraph}{5}%
                 {\parindent}%
                 {3.25ex plus 1ex minus .2ex}%
                 {-1em}%
                 {\STB@headfamily\normalsize\bfseries}}

% =========================================================
% Captions (STB Style)
% =========================================================
\setlength{\abovecaptionskip}{0.5\baselineskip}
\setlength{\belowcaptionskip}{0.5\baselineskip}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\small{\bfseries #1:} #2}% % Caption format: Figure 1.1: Text, bold label
  \ifdim \wd\@tempboxa >\hsize
    \unhbox\@tempboxa\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}% % Center if short
  \fi
  \vskip\belowcaptionskip}


% =========================================================
% Front matter placeholders
% =========================================================
\newcommand{\submitted}[1]{\gdef\@submitted{#1}}
\newcommand{\advisor}[1]{\gdef\@advisor{#1}}
\newcommand{\coadvisor}[1]{\gdef\@coadvisor{#1}}
\newcommand{\cocoadvisor}[1]{\gdef\@cocoadvisor{#1}}
\newcommand{\dedicationfile}[1]{\gdef\@dedicationfile{#1}}
\newcommand{\abstractfile}[1]{\gdef\@abstractfile{#1}}
\newcommand{\prefacefile}[1]{\gdef\@prefacefile{#1}}
\newcommand{\declarationplagfile}[1]{\gdef\@declarationplagfile{#1}}
\newcommand{\declarationpubfile}[1]{\gdef\@declarationpubfile{#1}}
\newcommand{\acknowledgementsfile}[1]{\gdef\@acknowledgementsfile{#1}}
\newcommand{\symbolsfile}[1]{\gdef\@symbolsfile{#1}}
\newcommand{\acronymsfile}[1]{\gdef\@acronymsfile{#1}}

\def\@submitted{}\def\@author{}\def\@advisor{}\def\@coadvisor{}\def\@cocoadvisor{}
\def\@dedicationfile{}\def\@abstractfile{}\def\@prefacefile{}\def\@declarationplagfile{}
\def\@declarationpubfile{}\def\@acknowledgementsfile{}
\def\@symbolsfile{}
\def\@acronymsfile{}

% -----------------------------
% Title / Author placeholders
% -----------------------------
\def\@title{}             % thesis title
%\newcommand{\title}[1]{\gdef\@title{#1}}

\def\@author{}            % author name
\renewcommand{\author}[1]{\gdef\@author{#1}}
% =========================================================
% Page-preserving wrapper for lists
% =========================================================
\makeatletter
\newcommand{\preservelist}[2]{%
  %\phantomsection
  \addcontentsline{toc}{section}{#1}%
  #2%
}
\makeatother

% =========================================================
% Glossaries
% =========================================================
\usepackage[nonumberlist,acronym,toc]{glossaries}
\makenoidxglossaries

% -----------------------------
% Pre-title page
% -----------------------------
\newcommand{\pretitlepage}{%
  \thispagestyle{empty}%
  \begin{center}
  \vspace*{2cm}
     {\LARGE \bfseries Development and Internal Validation of the H-potential:}\\
     {\Large  A Multidimensional Health Metric for Predicting Health Outcomes}\\
      \vspace*{1cm}
     {\LARGE Ronald Rethabile Malamule Mahomane}
    
    % Optional logo
    \ifx\@ThesisLogo\empty\else
      \includegraphics[width=0.70\paperwidth]{Figures/university-of-the-witwatersrand-johannesburg-seeklogo.png}\\[1em]
    \fi
    
    
    {\Large November 2025}
  \end{center}%
  \clearpage
}

% -----------------------------
% Main title page (hard-coded, with monochrome watermark)
% -----------------------------
\newcommand{\maketitlepage}{%
  \thispagestyle{empty}%
  
  % TikZ watermark
  \begin{tikzpicture}[remember picture, overlay]
    \node[opacity=0.08, inner sep=0] at (current page.center){
      \includegraphics[width=0.75\paperwidth]{Figures/university-of-the-witwatersrand-johannesburg-seeklogo.png}
    };
  \end{tikzpicture}%

  % Vertical centering
  \vspace{0.5cm}
  \begin{center}
    % Title & author
    {\LARGE \bfseries Development and Internal Validation of the H-potential:}\\
     {\Large  A Multidimensional Health Metric for Predicting Health Outcomes}\\[9cm]
    {Ronald Rethabile Malamule Mahomane}\\

    % Supervisors
    Supervisor: Professor Johnny Mahlangu\\
    Co-Supervisor: Professor Samuel Nuugulu\\
    Co-Supervisor: Doctor Farai Manwa\\[1em]

   % Degree & University
    Submitted in fulfilment of the requirements for the degree of\\
    \textit{Doctor of Philosophy}\\
    \textit{Department of Molecular Medicine and Haematology}\\
    \textit{School of Pathology}\\
    \textit{Faculty of Health Sciences}\\
    University of the Witwatersrand, Johannesburg\\[0.5em]

    % Submission date
    November 2025
  \end{center}
  \vspace*{\fill}
  \clearpage
}

% -----------------------------
% Post-title page
% -----------------------------
\newcommand{\posttitlepage}{%
  \thispagestyle{empty}%
  \vspace*{\fill}\begin{center}\end{center}\vspace*{\fill}\clearpage
}

% =========================================================
% Front matter sections
% =========================================================
\makeatletter
\newcommand{\makeabstract}{%
  \clearpage\thispagestyle{plain}%
  \preservelist{Abstract}{%
    {\Large\bfseries Abstract}\par\vspace{1em}%
    \IfFileExists{\@abstractfile}{\input{\@abstractfile}}{}%
  }%
  \clearpage
}

\newcommand{\makepreface}{%
  \clearpage\thispagestyle{plain}%
  \preservelist{Preface}{%
    {\Large\bfseries Preface}\par\vspace{1em}%
    \IfFileExists{\@prefacefile}{\input{\@prefacefile}}{}%
  }%
  \clearpage
}

\newcommand{\makedeclplag}{%
  \clearpage\thispagestyle{plain}%
  \preservelist{Declaration 1 - Plagiarism}{%
    {\Large\bfseries Candidate's Declaration}\par\vspace{1em}%
    \IfFileExists{\@declarationplagfile}{\input{\@declarationplagfile}}{}%
  }%
  \clearpage
}

\newcommand{\makedeclpub}{%
  \clearpage\thispagestyle{plain}%
  \preservelist{Declaration 2 - Publication}{%
    {\Large\bfseries Candidate's Declaration of Originality}\par\vspace{1em}%
    \IfFileExists{\@declarationpubfile}{\input{\@declarationpubfile}}{}%
  }%
  \clearpage
}

\newcommand{\makededication}{%
  \clearpage\thispagestyle{plain}%
  \IfFileExists{\@dedicationfile}{%
    \preservelist{Dedication}{%
      {\Large\bfseries Dedication
      %results
      }\par\vspace{1em}%
      \input{\@dedicationfile}%
    }%
  }{}%
  \clearpage
}

\newcommand{\makeacknowledgements}{%
  \clearpage\thispagestyle{plain}%
  \preservelist{Acknowledgements}{%
    {\Large\bfseries Acknowledgements}\par\vspace{1em}%
    \IfFileExists{\@acknowledgementsfile}{\input{\@acknowledgementsfile}}{}%
  }%
  \clearpage
}

% =========================================================
% List of Symbols / Abbreviations
% =========================================================

% Commands to set the files
\newcommand{\listsofsymbolsfile}[1]{\renewcommand{\@symbolsfile}{#1}}
\newcommand{\listofabbreviationsfile}[1]{\renewcommand{\@acronymsfile}{#1}}

% --- List of Symbols ---
\newcommand{\makelos}{%
  \clearpage
  \thispagestyle{plain}%
  \phantomsection
  \chapter*{List of Symbols}%
  \addcontentsline{toc}{chapter}{List of Symbols}%
  \IfFileExists{\@symbolsfile}{\input{\@symbolsfile}}{%
    \vspace{1em}\noindent\textit{No symbols file found.}%
  }%
  \clearpage
}

% --- List of Abbreviations ---
\newcommand{\makeloa}{%
  \clearpage
  \thispagestyle{plain}%
  \phantomsection
  \chapter*{List of Abbreviations}%
  \addcontentsline{toc}{chapter}{List of Abbreviations}%
  \IfFileExists{\@acronymsfile}{\input{\@acronymsfile}}{%
    \vspace{1em}\noindent\textit{Abbreviations file not found: \@acronymsfile}%
  }%
  \clearpage
}

% =========================================================
% Figures, Tables, Worked Examples
% =========================================================
\let\oldlistoffigures\listoffigures
\renewcommand{\listoffigures}{\oldlistoffigures\clearpage}
\let\oldlistoftables\listoftables
\renewcommand{\listoftables}{\oldlistoftables\clearpage}

% Set dotted line style for LoF/LoT/LoE from stb-thesis.cls
\renewcommand*\l@figure{\@dottedtocline{1}{0em}{3em}}
\let\l@table\l@figure
% This needs to be done *after* the LoadClass
\AtBeginDocument{%
    \let\l@workedexample\l@figure
}

% -----------------------------
% Counter for worked examples
% -----------------------------
\newcounter{workedexample}[chapter]
\renewcommand{\theworkedexample}{\thechapter.\arabic{workedexample}}

% -----------------------------
% Environment for worked examples
% -----------------------------
\newenvironment{workedexample}[2][]{%
  \refstepcounter{workedexample}% increment counter
  \par\medskip
  \noindent\textbf{Worked Example \theworkedexample: #2}\par
  % Add entry to the list of worked examples
  \addcontentsline{workedexample}{workedexample}{\protect\numberline{\theworkedexample}#2}%
  % Optional label for cross-referencing
  \ifx\relax#1\relax\else\label{ex:#1}\fi
  \medskip
}{%
  \par\medskip
}

% -----------------------------
% List of Worked Examples
% -----------------------------
\makeatletter
\newcommand{\listofworkedexamples}{%
  \clearpage
  \phantomsection
  \addcontentsline{toc}{chapter}{List of Worked Examples}%
  {\Large\bfseries List of Worked Examples}\par\vspace{1em}%
  \begingroup
    \let\l@workedexample\l@figure
    \@starttoc{workedexample}%
  \endgroup
  \clearpage
}
\makeatother

% =========================================================
% Theorems, Lemmas, etc.
% =========================================================
\declaretheorem[numberwithin=chapter]{theorem}
\declaretheorem[sibling=theorem]{lemma}
\declaretheorem[sibling=theorem]{proposition}
\declaretheorem[sibling=theorem]{corollary}
\declaretheorem[numberwithin=chapter,style=definition]{definition}
\declaretheorem[numberwithin=chapter,style=remark]{remark}
\declaretheorem[numberwithin=chapter,style=definition]{derivation}
\declaretheorem[numberwithin=chapter,style=definition]{axiom}
\declaretheorem[numberwithin=chapter,style=definition]{assumption}

\renewcommand{\listtheoremname}{List of Theorems, Definitions and Propositions}
\newcommand{\makeloth}{%
  \clearpage
  \thispagestyle{plain}%
  \phantomsection
  % thmtools prints its own heading via \listtheoremname; do not add a second.
  \addcontentsline{toc}{chapter}{\listtheoremname}%
  \listoftheorems[ignoreall,show={theorem,lemma,proposition,corollary,definition,axiom,assumption}]%
  \clearpage
}

% =========================================================
% Front/Back/Main Page Style
% =========================================================
% frontmatter: plain (footer number), main/backmatter: headings (running head + footer number)
\pretocmd{\frontmatter}{\pagestyle{plain}\pagenumbering{roman}}{}{}
\pretocmd{\mainmatter}{\pagestyle{headings}\pagenumbering{arabic}}{}{}
\pretocmd{\backmatter}{\pagestyle{headings}}{}{}

% Patch ToC to remove phantom pages
\makeatletter
\patchcmd{\tableofcontents}{\clearpage}{}{}{}
\patchcmd{\tableofcontents}{\cleardoublepage}{}{}{}
\apptocmd{\tableofcontents}{\par\vspace{-1em}}{}{}
\makeatother

% =========================================================
% Front-matter list command
% =========================================================
\newcommand{\frontmatterlists}{%
  \makeloa
  \makelos
  \makeloth
  \listoffigures
  \listoftables
  \listofworkedexamples
}

% =========================================================
% Begin Command
% =========================================================
\newcommand{\begincmd}{%
  % \doublespacing % REMOVED to match STB-Thesis style
  \frontmatter
  \pretitlepage
  \maketitlepage
  \posttitlepage
% --- START PAGE NUMBERING HERE ---
\clearpage
\pagenumbering{roman}   % Roman numerals: i, ii, iii...
\setcounter{page}{1}    % Abstract = page i

  \makeabstract
  \makepreface
  \makedeclplag
  \makedeclpub
  \makededication
  \makeacknowledgements

  % TOC
  \begingroup\let\cleardoublepage\relax\tableofcontents\endgroup

  % All front-matter lists
  \frontmatterlists

  \mainmatter
}

\AtBeginDocument{\begincmd}

\endinput
