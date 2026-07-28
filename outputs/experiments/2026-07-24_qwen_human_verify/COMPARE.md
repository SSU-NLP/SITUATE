# Qwen v2 인간검증 (①B human 라벨 재사용, 재라벨 0)

> 같은 repcorpus 100문장. 사람 y/n(비중립/중립 블라인드) 그대로. Qwen3.6-35B-A3B v2 라벨 얹어 대조.
> **밑에 [사람 · deepseek · Qwen] 나란히** — 셋이 갈리는 곳이 v3 재설계 후보.
> 사람 'no'=중립(unspecified/none/absent 등이 정답) · 'yes'=비중립(listed 매치). model 'neutral'=중립토큰 라벨.

## 축별 요약 — human vs Qwen (deepseek 대비)

| 축 | n | human-Qwen 일치 | Qwen over | Qwen under | (참고)human-deepseek 일치 |
|---|---|---|---|---|---|
| space | 20 | 14/20 (70%) | 3 | 3 | 15/20 (75%) |
| intentionality | 20 | 17/20 (85%) | 0 | 3 | 18/20 (90%) |
| time | 20 | 18/20 (90%) | 0 | 2 | 19/20 (95%) |
| protagonist | 20 | 17/20 (85%) | 1 | 2 | 19/20 (95%) |
| causality | 20 | 15/20 (75%) | 1 | 4 | 14/20 (70%) |

## 문장별 [사람 · deepseek · Qwen] (🔴=human↔Qwen 불일치)

### space (scored=location, 중립=unspecified)

- `space:fam_doyle_speckled:166` 🔴over  사람=**no** · ds=`julia_bedroom` · qwen=`roylott_estate`
  > When he reached my sister’s side she was unconscious, and though he poured brandy down her throat and sent for medical a
- `space:fam_london_fire:214` 🔴over  사람=**no** · ds=`yukon_trail` · qwen=`yukon_trail`
  > But it was surprising, the rapidity with which his cheeks and nose were freezing.
- `space:obs_25744:18` 🔴under  사람=**yes** · ds=`unspecified` · qwen=`unspecified`
  > Sanchez touched a control on the arm of her command chair.
- `space:obs_74140:140` 🔴over  사람=**no** · ds=`unspecified` · qwen=`post_office`
  > Steve Martin, that runs the post office, he come out here to go into the real-estate business, and he did too, and he’s 
- `space:obs_77176:14` 🔴under  사람=**yes** · ds=`unspecified` · qwen=`unspecified`
  > A glance at the map pinned down in front of him showed that, when vertically above a certain road crossing, he would be 
- `space:obs_77176:62` 🔴under  사람=**unclear** · ds=`unspecified` · qwen=`unspecified`
  > The control required the use of both hands and the rush of wind made the paper difficult to keep in place, but he accomp
- `space:fam_bierce_owlcreek:31`  사람=**yes** · ds=`owl_creek_bridge` · qwen=`Owl_Creek_bridge`
  > At a signal from the former the latter would step aside, the plank would tilt and the condemned man go down between two 
- `space:fam_bierce_owlcreek:8`  사람=**yes** · ds=`owl_creek_bridge` · qwen=`Owl_Creek_bridge`
  > It did not appear to be the duty of these two men to know what was occurring at the center of the bridge; they merely bl
- `space:fam_doyle_speckled:66`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > “Alas!” replied our visitor, “the very horror of my situation lies in the fact that my fears are so vague, and my suspic
- `space:fam_gilman_wallpaper:100`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > I never saw such ravages as the children have made here.
- `space:obs_25744:71`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > Normal food made them violently ill, and strong spices caused anaphylactic shock, usually fatal.
- `space:obs_28922:20`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > If things get worse, get a divorce.
- `space:obs_28922:98`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > He raised Helen's soft brown hair and opened a small panel in the back of her head.
- `space:obs_37511:42`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > Also Manufacturers of every description of ACCOUNT BOOKS, MEMORANDUMS and PASS BOOKS, a large Stock of which is constant
- `space:obs_74140:70`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > When things happen to people, things happen to them, and that’s all there is to it. * * * * * They summoned courage, fin
- `space:obs_74162:128`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > "Amelia doesn't know.
- `space:obs_74162:17`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > And I'll lose Amelia forever._ That was the thing he could not bear.
- `space:obs_74162:199`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > "They told me you were recalcitrant.
- `space:obs_74162:99`  사람=**no** · ds=`unspecified` · qwen=`unspecified`
  > Why did you come back?
- `space:obs_77176:115`  사람=**unclear** · ds=`trenches` · qwen=`enemy_lines`
  > He zigzagged, climbed, dived and climbed again; he resorted to every maneuver he had ever been taught and that his old c

### intentionality (scored=active_goal, 중립=none)

- `intentionality:fam_bierce_owlcreek:130` 🔴under  사람=**unclear** · ds=`shoot_escaping_prisoner` · qwen=`none`
  > Suddenly he heard a sharp report and something struck the water smartly within a few inches of his head, spattering his 
- `intentionality:fam_gilman_wallpaper:238` 🔴under  사람=**unclear** · ds=`find_out_wallpaper_secret` · qwen=`none`
  > She didn’t know I was in the room, and when I asked her in a quiet, a very quiet voice, with the most restrained manner 
- `intentionality:obs_74140:125` 🔴under  사람=**unclear** · ds=`none` · qwen=`none`
  > Ducky likes to cook, but he can’t, and I’m a good cook, but I wont, so we figured we’d better get in a neutral party.
- `intentionality:fam_bierce_owlcreek:164`  사람=**no** · ds=`none` · qwen=`none`
  > He dug his fingers into the sand, threw it over himself in handfuls and audibly blessed it.
- `intentionality:fam_bierce_owlcreek:34`  사람=**no** · ds=`none` · qwen=`none`
  > He looked a moment at his “unsteadfast footing,” then let his gaze wander to the swirling water of the stream racing mad
- `intentionality:fam_doyle_speckled:177`  사람=**yes** · ds=`investigate_case` · qwen=`investigate_death`
  > And what conclusions did the coroner come to?
- `intentionality:fam_doyle_speckled:81`  사람=**no** · ds=`none` · qwen=`none`
  > “When Dr.
- `intentionality:fam_doyle_speckled:83`  사람=**no** · ds=`none` · qwen=`none`
  > Stoner, the young widow of Major-General Stoner, of the Bengal Artillery.
- `intentionality:fam_gilman_wallpaper:198`  사람=**no** · ds=`none` · qwen=`none`
  > “Why, how can I, dear?
- `intentionality:fam_jewett_heron:28`  사람=**no** · ds=`none` · qwen=`none`
  > There was a stirring in the great boughs overhead.
- `intentionality:fam_jewett_heron:86`  사람=**yes** · ds=`none` · qwen=`collect_birds`
  > “I am making a collection of birds myself.
- `intentionality:fam_twain_frog:35`  사람=**no** · ds=`none` · qwen=`none`
  > Leonidas W.
- `intentionality:obs_25744:199`  사람=**no** · ds=`none` · qwen=`none`
  > Thompson was unwillingly impressed; an Imperial Count's time was worth considerably more than a Marine Captain's.
- `intentionality:obs_25744:218`  사람=**unclear** · ds=`enjoy_leave` · qwen=`quarter_team_in_system_palace`
  > And we won't get too many chances at living in a System Palace; pass the word to take full advantage of it.
- `intentionality:obs_25744:63`  사람=**no** · ds=`none` · qwen=`none`
  > Not that he really had to give them orders any more, he thought.
- `intentionality:obs_74140:217`  사람=**no** · ds=`none` · qwen=`none`
  > Returning, dog-tired, at dark, they found him on the bench in the kitchen, sitting perfectly still, eyes straight ahead,
- `intentionality:obs_74162:154`  사람=**no** · ds=`none` · qwen=`none`
  > "And you, Doctor, must be seven or eight years younger.
- `intentionality:obs_77176:100`  사람=**no** · ds=`none` · qwen=`none`
  > Now he experienced a wonderful exhilaration.
- `intentionality:obs_77176:101`  사람=**yes** · ds=`die_gloriously` · qwen=`die_gloriously`
  > Yes, he would die, but willingly and usefully!
- `intentionality:obs_77176:78`  사람=**no** · ds=`none` · qwen=`none`
  > God!

### time (scored=temporal_shift_marker, 중립=absent)

- `time:fam_jewett_heron:139` 🔴under  사람=**yes** · ds=`absent` · qwen=`absent`
  > First she must mount the white oak tree that grew alongside, where she was almost lost among the dark branches and the g
- `time:fam_twain_frog:2` 🔴under  사람=**yes** · ds=`sequence` · qwen=`absent`
  > THEN CLAWED BACK INTO A CIVILIZED LANGUAGE ONCE MORE BY PATIENT, UNREMUNERATED TOIL.
- `time:fam_bierce_owlcreek:190`  사람=**no** · ds=`absent` · qwen=`absent`
  > How softly the turf had carpeted the untraveled avenue—he could no longer feel the roadway beneath his feet!
- `time:fam_gilman_wallpaper:65`  사람=**no** · ds=`absent` · qwen=`absent`
  > Such a dear baby!
- `time:fam_jewett_heron:51`  사람=**no** · ds=`absent` · qwen=`absent`
  > Sylvia kept an awed silence; she knew by instinct that her grandmother did not comprehend the gravity of the situation.
- `time:fam_london_fire:118`  사람=**no** · ds=`absent` · qwen=`absent`
  > It certainly was cold.
- `time:fam_london_fire:227`  사람=**no** · ds=`absent` · qwen=`absent`
  > He should have built it in the open.
- `time:fam_london_fire:30`  사람=**no** · ds=`absent` · qwen=`absent`
  > He knew that at fifty below spittle crackled on the snow, but this spittle had crackled in the air.
- `time:fam_twain_frog:16`  사람=**no** · ds=`absent` · qwen=`absent`
  > Without further introduction, the Jumping Frog, as I originally wrote it, was as follows [after it will be found the Fre
- `time:fam_twain_frog:51`  사람=**no** · ds=`absent` · qwen=`absent`
  > He give Smiley a look, as much as to say his heart was broke, and it was his fault, for putting up a dog that hadn't no 
- `time:obs_25744:194`  사람=**no** · ds=`absent` · qwen=`absent`
  > "So be it, then, Captain, although I had hoped you would be more reasonable.
- `time:obs_25744:49`  사람=**no** · ds=`absent` · qwen=`absent`
  > "What's that?
- `time:obs_28922:0`  사람=**no** · ds=`absent` · qwen=`absent`
  > Produced by Greg Weeks, Stephen Blundell and the Online Distributed Proofreading Team at https://www.pgdp.net _Richard R
- `time:obs_28922:35`  사람=**no** · ds=`absent` · qwen=`absent`
  > My face blanched.
- `time:obs_28922:93`  사람=**no** · ds=`absent` · qwen=`absent`
  > We did it together.
- `time:obs_74140:13`  사람=**no** · ds=`absent` · qwen=`absent`
  > But the postmaster had returned, with the popping suddenness of a prairie-dog, to his hole of an office, and Dog saw tha
- `time:obs_74140:55`  사람=**no** · ds=`absent` · qwen=`absent`
  > Threatened with T.B., I reckon, though she don’t come right out with it.
- `time:obs_74162:173`  사람=**no** · ds=`absent` · qwen=`absent`
  > He stared into keen blue eyes.
- `time:obs_74162:57`  사람=**no** · ds=`absent` · qwen=`absent`
  > There was a swift brook in the woods beyond his cultivated acres.
- `time:obs_74162:7`  사람=**no** · ds=`absent` · qwen=`absent`
  > And a great furred honeybee circled and dipped, touching the vivid azaleas, drinking the heart of the iris, and swiftly 

### protagonist (scored=perspective, 중립=none)

- `protagonist:fam_gilman_wallpaper:209` 🔴over  사람=**no** · ds=`none` · qwen=`first`
  > The color is hideous enough, and unreliable enough, and infuriating enough, but the pattern is torturing.
- `protagonist:fam_london_fire:184` 🔴under  사람=**yes** · ds=`third` · qwen=`none`
  > He knew there must be no failure.
- `protagonist:obs_74162:5` 🔴under  사람=**yes** · ds=`third` · qwen=`none`
  > Evelyn Goldstein has captured the tragedy of the not-quite-human with a deftness extraordinary._ This golden day was pol
- `protagonist:fam_gilman_wallpaper:15`  사람=**yes** · ds=`first` · qwen=`first`
  > Personally, I believe that congenial work, with excitement and change, would do me good.
- `protagonist:fam_gilman_wallpaper:156`  사람=**yes** · ds=`first` · qwen=`first`
  > I tried to have a real earnest reasonable talk with him the other day, and tell him how I wish he would let me go and ma
- `protagonist:fam_gilman_wallpaper:77`  사람=**yes** · ds=`first` · qwen=`first`
  > Out of one window I can see the garden, those mysterious deep-shaded arbors, the riotous old-fashioned flowers, and bush
- `protagonist:fam_jewett_heron:120`  사람=**yes** · ds=`third` · qwen=`third`
  > She grieved because the longed-for white heron was elusive, but she did not lead the guest, she only followed, and there
- `protagonist:fam_london_fire:106`  사람=**yes** · ds=`third` · qwen=`third`
  > Once again, however, he had a close call; and once, suspecting danger, he compelled the dog to go on in front.
- `protagonist:fam_twain_frog:50`  사람=**yes** · ds=`third` · qwen=`third`
  > Smiley always come out winner on that pup, till he harnessed a dog once that didn't have no hind legs, because they'd be
- `protagonist:obs_25744:201`  사람=**yes** · ds=`first` · qwen=`first`
  > "By the Count's leave, I would like to return to my people.
- `protagonist:obs_25744:211`  사람=**yes** · ds=`third` · qwen=`third`
  > He described his conversation with the Count, then scowled, knowing Audra wouldn't take it as aimed at her.
- `protagonist:obs_25744:67`  사람=**yes** · ds=`third` · qwen=`third`
  > He started with the oldest, published about six weeks ago, discovering that the Archbishop's basic facts were accurate.
- `protagonist:obs_28922:42`  사람=**yes** · ds=`first` · qwen=`first`
  > I felt short of breath.
- `protagonist:obs_28922:50`  사람=**yes** · ds=`first` · qwen=`first`
  > But both of us had a habit of stopping at a club on the outskirts of town and we met there often.
- `protagonist:obs_37511:43`  사람=**yes** · ds=`none` · qwen=`third`
  > Their Assortment of SCHOOL AND MISCELLANEOUS BOOKS, and of Foreign and Domestic STATIONERY, is very complete, to the ins
- `protagonist:obs_74140:45`  사람=**yes** · ds=`third` · qwen=`third`
  > Ducky slumped again.
- `protagonist:obs_74140:76`  사람=**yes** · ds=`third` · qwen=`third`
  > Dog went to buy some white soap, and then, remembering something, bought a dozen cakes of laundry soap as well.
- `protagonist:obs_74162:165`  사람=**yes** · ds=`first` · qwen=`first`
  > "You see, I have two children.
- `protagonist:obs_77176:21`  사람=**no** · ds=`none` · qwen=`none`
  > And again!
- `protagonist:obs_77176:79`  사람=**yes** · ds=`third` · qwen=`third`
  > If he only had an observer with him, or if he could only leave the control for a few seconds!

### causality (scored=aspect, 중립=ongoing)

- `causality:fam_doyle_speckled:157` 🔴under  사람=**yes** · ds=`completive` · qwen=`ongoing`
  > By the light of the corridor-lamp I saw my sister appear at the opening, her face blanched with terror, her hands gropin
- `causality:fam_jewett_heron:38` 🔴under  사람=**yes** · ds=`ongoing` · qwen=`ongoing`
  > She did not dare to look boldly at the tall young man, who carried a gun over his shoulder, but she came out of her bush
- `causality:fam_london_fire:236` 🔴under  사람=**unclear** · ds=`ongoing` · qwen=`ongoing`
  > Where it had burned was a mantle of fresh and disordered snow.
- `causality:obs_25744:203` 🔴under  사람=**yes** · ds=`ongoing` · qwen=`ongoing`
  > But please remember, I am not doing this out of hostility; I do what I do only because I think it best for my own people
- `causality:obs_74140:185` 🔴over  사람=**no** · ds=`completive` · qwen=`completive`
  > Spud shook his head.
- `causality:fam_doyle_speckled:52`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > I have no one to turn to—none, save only one, who cares for me, and he, poor fellow, can be of little aid.
- `causality:fam_gilman_wallpaper:207`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > He thought I was asleep first, but I wasn’t,—I lay there for hours trying to decide whether that front pattern and the b
- `causality:fam_gilman_wallpaper:61`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > It does weigh on me so not to do my duty in any way!
- `causality:fam_gilman_wallpaper:8`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > John is a physician, and perhaps—(I would not say it to a living soul, of course, but this is dead paper and a great rel
- `causality:fam_jewett_heron:163`  사람=**yes** · ds=`completive` · qwen=`completive`
  > The child gives a long sigh a minute later when a company of shouting cat-birds comes also to the tree, and vexed by the
- `causality:fam_jewett_heron:63`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > There’s good pasturing for geese just below here towards the ma’sh.
- `causality:fam_london_fire:104`  사람=**no** · ds=`completive` · qwen=`ongoing`
  > In the course of the next two hours he came upon several similar traps.
- `causality:fam_twain_frog:41`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > If he even see a straddle-bug start to go anywheres, he would bet you how long it would take him to get to--to wherever 
- `causality:obs_25744:165`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > The Count looked pleased.
- `causality:obs_25744:182`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > "And that you are trying to avoid thinking about it, because you want it as well.
- `causality:obs_74140:105`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > “I want to go back.
- `causality:obs_74162:13`  사람=**unclear** · ds=`ongoing` · qwen=`inceptive`
  > He raised haunted eyes to the hills beyond.
- `causality:obs_74162:145`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > Can you cut yourself and not feel pain?
- `causality:obs_74162:59`  사람=**yes** · ds=`completive` · qwen=`completive`
  > The basket he broke in his strong brown hands and sent the pieces after the flowers.
- `causality:obs_77176:0`  사람=**no** · ds=`ongoing` · qwen=`ongoing`
  > IN MID-AIR By R.

