# Gate 0 — GutenQA × Phase-1 앵커 오염 체크

- GutenQA 도서: **100편**
- 앵커(D-B15) 고유 PG ID: **3042개** / 10672 pieces
- 🔴 **오염(교집합): 2편**
- 자동매칭 실패 → 사람 판정으로 확정: 5편 · 끝내 미매칭(UNMATCHED): 0편 · 저자 불일치(title-only): 0편

## 🔴 오염된 책 (평가에서 제외 대상)

| Book ID | Book Name | PG ID | 앵커 조각 수 | 앵커 문장 수 |
|---|---|---|---|---|
| 20 | Dubliners_-_James_Joyce | [2814] | 9 | 3,531 |
| 62 | The_King_in_Yellow_-_Robert_W._Chambers | [8492] | 15 | 4,327 |

두 권 모두 **단편집**이다 — 앵커 빌더의 `"short stories" in Subjects` 필터를 정상 통과했고, 저자(Joyce / Chambers)가 `EXCL_AUTH` 14명에 없다. 예측대로 장편은 한 편도 새지 않았고, 샌 것은 단편집뿐이다.

- **Dubliners_-_James_Joyce** (PG [2814]) 앵커 수록 조각: `THE SISTERS`, `AN ENCOUNTER`, `AFTER THE RACE`, `TWO GALLANTS`, `THE BOARDING HOUSE`, `A LITTLE CLOUD`, `A PAINFUL CASE`, `ROYAL EXCHANGE WARD`, `THE DEATH OF PARNELL`
- **The_King_in_Yellow_-_Robert_W._Chambers** (PG [8492]) 앵커 수록 조각: `THE REPAIRER OF REPUTATIONS`, `HAWBERK, ARMOURER.`, `“THE IMPERIAL DYNASTY OF AMERICA.”`, `“MR. WILDE,`, `REPAIRER OF REPUTATIONS.`, `THE MASK`, `IN THE COURT OF THE DRAGON`, `THE YELLOW SIGN`, `THE DEMOISELLE D’YS`, `THE STREET OF THE FOUR WINDS`, `THE STREET OF THE FIRST SHELL`, `THE STREET OF OUR LADY OF THE FIELDS`, `|  RICHARD OSBORNE ELLIOTT  |`, `RUE BARRÉE`, `RUE BARRÉE.`

## 사람 판정으로 확정한 매핑 (자동 fuzzy 채택 금지 규칙에 따름)

악센트(Misérables/Vallière)·부제·분책 때문에 문자열 자동매칭이 실패한 건들이다. `MANUAL_TITLE_MAP` 에 코드로 못박아 재실행 시에도 동일 결론이 나온다.

| Book ID | Book Name | 확정 PG ID | 앵커? |
|---|---|---|---|
| 27 | Incidents_in_the_Life_of_a_Slave_Girl_-_Harriet_A._Jacobs | [11030] | ✅ no |
| 29 | Les_Miserables_-_Victor_Hugo | [135] | ✅ no |
| 31 | Louise_de_la_Valliere_-_Alexandre_Dumas | [2710] | ✅ no |
| 49 | The_Adventures_of_Tom_Sawyer_-_Mark_Twain | [74, 7193, 7194, 7195, 7196] | ✅ no |
| 58 | The_Extraordinary_Adventures_of_Arsene_Lupin_-_Maurice_Leblanc | [6133] | ✅ no |

## 전체 100편

| ID | Book Name | PG ID | 앵커? | 신뢰도 | EXCL_AUTH 사전배제 |
|---|---|---|---|---|---|
| 0 | Adventures_of_Huckleberry_Finn_-_Mark_Twain | 76 | ✅ no | exact(title+author) | Twain |
| 1 | Alice's_Adventures_in_Wonderland_-_Lewis_Carroll | 11,928,19033… | ✅ no | exact(title+author) | — |
| 2 | Anna_Karenina_-_Leo_Tolstoy | 1399,13214,49487 | ✅ no | exact(title+author) | — |
| 3 | Anne_of_Green_Gables_-_L._M._Montgomery | 45,64365 | ✅ no | exact(title+author) | — |
| 4 | Around_the_World_in_Eighty_Days_-_Jules_Verne | 103 | ✅ no | exact(title+author) | — |
| 5 | A_Christmas_Carol_-_Charles_Dickens | 19337,24022,30368… | ✅ no | exact(title+author) | — |
| 6 | A_Connecticut_Yankee_in_King_Arthur's_Court_-_Mark_Twain | 86 | ✅ no | exact(title+author) | Twain |
| 7 | A_Crystal_Age_-_W_H_Hudson | 7401 | ✅ no | exact(title+author) | — |
| 8 | A_Legend_Of_Montrose_-_Sir_Walter_Scott | 1461 | ✅ no | exact(title+author) | — |
| 9 | A_Portrait_of_the_Artist_as_a_Young_Man_-_James_Joyce | 4217 | ✅ no | exact(title+author) | — |
| 10 | A_Room_with_a_View_-_E._M._Forster | 2641 | ✅ no | exact(title+author) | — |
| 11 | A_Study_in_Scarlet_-_Arthur_Conan_Doyle | 244 | ✅ no | exact(title+author) | Doyle, Arthur Conan |
| 12 | A_Tale_of_Two_Cities_-_Charles_Dickens | 98 | ✅ no | exact(title+author) | — |
| 13 | Beyond_Good_and_Evil_-_Friedrich_Wilhelm_Nietzsche | 4363 | ✅ no | exact(title+author) | — |
| 14 | Carmilla_-_Joseph_Sheridan_Le_Fanu | 10007 | ✅ no | exact(title+author) | — |
| 15 | Cranford_-_Elizabeth_Cleghorn_Gaskell | 394,57539 | ✅ no | exact(title+author) | — |
| 16 | Crime_and_Punishment_-_Fyodor_Dostoyevsky | 2554 | ✅ no | exact(title+author) | — |
| 17 | Daddy_takes_us_to_the_Garden_-_Howard_R._Garis | 14859 | ✅ no | exact(title+author) | — |
| 18 | David_Copperfield_-_Charles_Dickens | 766 | ✅ no | exact(title+author) | — |
| 19 | Dracula_-_Bram_Stoker | 345,45839 | ✅ no | exact(title+author) | — |
| 20 | Dubliners_-_James_Joyce | 2814 | 🔴 YES | exact(title+author) | — |
| 21 | Emma_-_Jane_Austen | 158 | ✅ no | exact(title+author) | — |
| 22 | Frankenstein_Or_The_Modern_Prometheus_-_Mary_Wollstonecraft_ | 84,41445,42324 | ✅ no | exact(title+author) | — |
| 23 | Great_Expectations_-_Charles_Dickens | 1400 | ✅ no | exact(title+author) | — |
| 24 | Grimms'_Fairy_Tales_-_Grimm_Brothers | 2591 | ✅ no | exact(title+author) | — |
| 25 | Gulliver's_Travels_into_Several_Remote_Nations_of_the_World_ | 829 | ✅ no | exact(title+author) | — |
| 26 | Heart_of_Darkness_-_Joseph_Conrad | 219,526 | ✅ no | exact(title+author) | — |
| 27 | Incidents_in_the_Life_of_a_Slave_Girl_-_Harriet_A._Jacobs | 11030 | ✅ no | manual(사람 판정) | — |
| 28 | Jane_Eyre_-_Charlotte_Bronte | 1260,16235 | ✅ no | exact(title+author) | Gilman, Charlotte |
| 29 | Les_Miserables_-_Victor_Hugo | 135 | ✅ no | manual(사람 판정) | — |
| 30 | Little_Women_-_Louisa_May_Alcott | 514,37106 | ✅ no | exact(title+author) | — |
| 31 | Louise_de_la_Valliere_-_Alexandre_Dumas | 2710 | ✅ no | manual(사람 판정) | — |
| 32 | Madame_Bovary_-_Gustave_Flaubert | 2413,14155,27575 | ✅ no | exact(title+author) | — |
| 33 | Meditations_-_Emperor_of_Rome_Marcus_Aurelius | 2680 | ✅ no | exact(title+author) | — |
| 34 | Middlemarch_-_George_Eliot | 145 | ✅ no | exact(title+author) | — |
| 35 | Moby_Dick_Or_The_Whale_-_Herman_Melville | 15,2489,2701 | ✅ no | exact(title+author) | Melville |
| 36 | My_Life_Volume_1_-_Richard_Wagner | 5197 | ✅ no | exact(title+author) | — |
| 37 | Narrative_of_the_Life_of_Frederick_Douglass,_an_American_Sla | 23 | ✅ no | exact(title+author) | — |
| 38 | Oliver_Twist_-_Charles_Dickens | 730,16023,46675… | ✅ no | exact(title+author) | — |
| 39 | Persuasion_-_Jane_Austen | 105,36777 | ✅ no | exact(title+author) | — |
| 40 | Peter_Pan_-_J._M._Barrie | 16,78131 | ✅ no | exact(title+author) | — |
| 41 | Pride_and_Prejudice_-_Jane_Austen | 1342,42671 | ✅ no | exact(title+author) | — |
| 42 | Riders_Of_The_Silences_-_Max_Brand | 9867,20044 | ✅ no | exact(title+author) | — |
| 43 | Second_Treatise_of_Government_-_John_Locke | 7370 | ✅ no | exact(title+author) | — |
| 44 | Sense_and_Sensibility_-_Jane_Austen | 161,21839 | ✅ no | exact(title+author) | — |
| 45 | Ten_Years_Later_-_Alexandre_Dumas | 1258,2681 | ✅ no | exact(title+author) | — |
| 46 | The_Adventures_of_Ferdinand_Count_Fathom_-_T._Smollett | 6759,6760,6761 | ✅ no | exact(title+author) | — |
| 47 | The_Adventures_of_Roderick_Random_-_T._Smollett | 4085 | ✅ no | exact(title+author) | — |
| 48 | The_Adventures_of_Sherlock_Holmes_-_Arthur_Conan_Doyle | 1661,48320 | ✅ no | exact(title+author) | Doyle, Arthur Conan |
| 49 | The_Adventures_of_Tom_Sawyer_-_Mark_Twain | 74,7193,7194… | ✅ no | manual(사람 판정) | Twain |
| 50 | The_Black_Tulip_-_Alexandre_Dumas | 965 | ✅ no | exact(title+author) | — |
| 51 | The_Blue_Castle_-_L._M._Montgomery | 67979 | ✅ no | exact(title+author) | — |
| 52 | The_Blue_Pavilions_-_Arthur_Thomas_Quiller_Couch | 19977 | ✅ no | exact(title+author) | Doyle, Arthur Conan |
| 53 | The_Brothers_Karamazov_-_Fyodor_Dostoyevsky | 28054 | ✅ no | exact(title+author) | — |
| 54 | The_Call_of_the_Wild_-_Jack_London | 215 | ✅ no | exact(title+author) | London, Jack |
| 55 | The_Confessions_of_St._Augustine_-_Bishop_of_Hippo_Saint_Aug | 3296,77585 | ✅ no | exact(title+author) | — |
| 56 | The_Count_of_Monte_Cristo_-_Alexandre_Dumas | 1184 | ✅ no | exact(title+author) | — |
| 57 | The_Enchanted_April_-_Elizabeth_Von_Arnim | 16389 | ✅ no | exact(title+author) | — |
| 58 | The_Extraordinary_Adventures_of_Arsene_Lupin_-_Maurice_Lebla | 6133 | ✅ no | manual(사람 판정) | — |
| 59 | The_Gambler_-_Fyodor_Dostoyevsky | 2197 | ✅ no | exact(title+author) | — |
| 60 | The_Great_Gatsby_-_F._Scott_Fitzgerald | 64317 | ✅ no | exact(title+author) | — |
| 61 | The_Hound_of_the_Baskervilles_-_Arthur_Conan_Doyle | 2852,3070 | ✅ no | exact(title+author) | Doyle, Arthur Conan |
| 62 | The_King_in_Yellow_-_Robert_W._Chambers | 8492 | 🔴 YES | exact(title+author) | — |
| 63 | The_Life_and_Adventures_of_Robinson_Crusoe_-_Daniel_Defoe | 521 | ✅ no | exact(title+author) | — |
| 64 | The_Man_in_the_Iron_Mask_-_Alexandre_Dumas | 2759 | ✅ no | exact(title+author) | — |
| 65 | The_Murder_of_Roger_Ackroyd_-_Agatha_Christie | 69087 | ✅ no | exact(title+author) | — |
| 66 | The_Murder_on_the_Links_-_Agatha_Christie | 58866 | ✅ no | exact(title+author) | — |
| 67 | The_Odyssey_-_Homer | 1727,3160 | ✅ no | exact(title+author) | — |
| 68 | The_Picture_of_Dorian_Gray_-_Oscar_Wilde | 174,4078,26740 | ✅ no | exact(title+author) | — |
| 69 | The_Prince_-_Niccolo_Machiavelli | 1232,57037 | ✅ no | exact(title+author) | — |
| 70 | The_Prince_and_the_Pauper_-_Mark_Twain | 1837 | ✅ no | exact(title+author) | Twain |
| 71 | The_Problems_of_Philosophy_-_Bertrand_Russell | 5827 | ✅ no | exact(title+author) | — |
| 72 | The_Prophet_-_Kahlil_Gibran | 58585 | ✅ no | exact(title+author) | — |
| 73 | The_Reign_of_Greed_-_Jose_Rizal | 10676 | ✅ no | exact(title+author) | — |
| 74 | The_Republic_-_Plato | 150,1497 | ✅ no | exact(title+author) | — |
| 75 | The_Romance_of_Lust_-_Anonymous | 30254 | ✅ no | exact(title+author) | — |
| 76 | The_Scarlet_Letter_-_Nathaniel_Hawthorne | 33,25344 | ✅ no | exact(title+author) | — |
| 77 | The_Sign_of_the_Four_-_Arthur_Conan_Doyle | 2097 | ✅ no | exact(title+author) | Doyle, Arthur Conan |
| 78 | The_Souls_of_Black_Folk_-_W._E._B._Du_Bois | 408 | ✅ no | exact(title+author) | — |
| 79 | The_Strange_Case_of_Dr._Jekyll_and_Mr._Hyde_-_Robert_Louis_S | 42,43 | ✅ no | exact(title+author) | — |
| 80 | The_Sun_Also_Rises_-_Ernest_Hemingway | 67138 | ✅ no | exact(title+author) | — |
| 81 | The_Three_Musketeers_-_Alexandre_Dumas | 1257 | ✅ no | exact(title+author) | — |
| 82 | The_Time_Machine_-_H._G._Wells | 35 | ✅ no | exact(title+author) | — |
| 83 | The_Trial_-_Franz_Kafka | 7849 | ✅ no | exact(title+author) | — |
| 84 | The_Turn_of_the_Screw_-_Henry_James | 209 | ✅ no | exact(title+author) | Henry, O. |
| 85 | The_Vicomte_de_Bragelonne_-_Alexandre_Dumas | 2609,18997 | ✅ no | exact(title+author) | — |
| 86 | The_War_of_the_Worlds_-_H._G._Wells | 36 | ✅ no | exact(title+author) | — |
| 87 | The_Wonderful_Wizard_of_Oz_-_L._Frank_Baum | 55,43936 | ✅ no | exact(title+author) | — |
| 88 | Thus_Spake_Zarathustra_-_Friedrich_Wilhelm_Nietzsche | 1998 | ✅ no | exact(title+author) | — |
| 89 | Treasure_Island_-_Robert_Louis_Stevenson | 120,27780 | ✅ no | exact(title+author) | — |
| 90 | Twenty_Thousand_Leagues_under_the_Sea_-_Jules_Verne | 164 | ✅ no | exact(title+author) | — |
| 91 | Twenty_Years_After_-_Alexandre_Dumas | 1259 | ✅ no | exact(title+author) | — |
| 92 | Ulysses_-_James_Joyce | 4300 | ✅ no | exact(title+author) | — |
| 93 | Uncle_Tom’s_Cabin_or_Life_among_the_Lowly_-_Harriet_Beecher_ | 203 | ✅ no | exact(title+author) | — |
| 94 | Walden,_and_On_The_Duty_Of_Civil_Disobedience_-_Henry_David_ | 205 | ✅ no | exact(title+author) | Henry, O. |
| 95 | War_and_Peace_-_Leo_Tolstoy | 2600 | ✅ no | exact(title+author) | — |
| 96 | Way_Of_The_Lawless_-_Max_Brand | 9903 | ✅ no | exact(title+author) | — |
| 97 | White_Nights_and_Other_Stories_-_Fyodor_Dostoyevsky | 36034 | ✅ no | exact(title+author) | — |
| 98 | Winnie_the_Pooh_-_A._A._Milne | 67098 | ✅ no | exact(title+author) | — |
| 99 | Wuthering_Heights_-_Emily_Bronte | 768 | ✅ no | exact(title+author) | — |
