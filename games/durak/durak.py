import numpy as np

suit_names = {0: "clubs", 1: "diamonds", 2: "hearts", 3: "spades"}
# suit_names = {0: "♣", 1: "♦", 2: "♥", 3: "♠"}
rank_names = {0: "6", 1: "7", 2: "8", 3: "9", 4: "10", 5: "jack", 6: "queen", 7: "king", 8: "ace"}

def get_id(rank, suit):
    return suit * 9 + rank
def get_rs(id_):
    s = np.floor(id_ / 9)
    r = id_ - 9 * s
    return r, s


class WorldDurak(object):
    def __init__(self):
        self.refresh()

    def refresh(self):
        self.status = "playerA_attack"
        self.deck = list(range(36))
        np.random.shuffle(self.deck)
        self.trump = get_rs(self.deck[0])[1]
        self.trump_id = self.deck[0]
        self.playerA = []
        self.playerB = []
        self.playerA += self.deck[-6:]
        self.deck = self.deck[:-6]
        self.playerB += self.deck[-6:]
        self.deck = self.deck[:-6]
        self.fight_attack = []
        self.fight_defense = []
        self.dump = []
        self.result = "continue"
    
    def drawPreferA(self):
        if len(self.playerA) < 6 and len(self.deck) > 0:
            draw = np.min([6 - len(self.playerA), len(self.deck)])
            self.playerA += self.deck[-draw:]
            self.deck = self.deck[:-draw]
        if len(self.playerB) < 6 and len(self.deck) > 0:
            draw = np.min([6 - len(self.playerB), len(self.deck)])
            self.playerB += self.deck[-draw:]
            self.deck = self.deck[:-draw]
    
    def drawPreferB(self):
        if len(self.playerB) < 6 and len(self.deck) > 0:
            draw = np.min([6 - len(self.playerB), len(self.deck)])
            self.playerB += self.deck[-draw:]
            self.deck = self.deck[:-draw]
        if len(self.playerA) < 6 and len(self.deck) > 0:
            draw = np.min([6 - len(self.playerA), len(self.deck)])
            self.playerA += self.deck[-draw:]
            self.deck = self.deck[:-draw]

    def checkValidForFight(self, card_id):
        if not self.fight_attack:
            return True
        r, _ = get_rs(card_id)
        for fight_card_id in self.fight_attack + self.fight_defense:
            fr, _ = get_rs(fight_card_id)
            if fr == r:
                return True

    def get_valid_moves(self):
        if "A" in self.status:
            this_player = self.playerA
        else:
            this_player = self.playerB
        moves = [0] * 37
        moves[36] = 1
        if "attack" in self.status:
            for action in range(36):
                if action in this_player and self.checkValidForFight(action):
                    moves[action] = 1
            if not self.fight_attack:
                moves[36] = 0
        elif "defense" in self.status: 
            for action in range(36):
                if action in this_player:
                    en_r, en_s = get_rs(self.fight_attack[-1])
                    r, s = get_rs(action)
                    if s == en_s and r <= en_r:
                        continue
                    if s != en_s and s != self.trump:
                        continue
                    moves[action] = 1
        else:
            raise RuntimeError("Unknown status for valid moves definition.")
        return moves

    def _act(self, action):
        if self.status == "playerA_attack":
            if action < 36:
                if action in self.playerA and self.checkValidForFight(action):
                    self.fight_attack.append(action)
                    self.playerA.remove(action)
                    self.status = "playerB_defense"
                    return "continue"
                else:
                    return "bad_move"
            if action == 36:
                if not self.fight_attack:
                    return "bad_move"
                self.dump.extend(self.fight_attack)
                self.dump.extend(self.fight_defense)
                self.fight_attack = []
                self.fight_defense = []
                self.drawPreferA()
                if len(self.playerA) == 0 and len(self.playerB) == 0:
                    return "stalemate"
                if len(self.playerA) == 0:
                    return "win_a"
                if len(self.playerB) == 0:
                    return "win_b"
                self.status = "playerB_attack"
                return "continue"
            raise Exception("This exception should never be raised.")

        if self.status == "playerB_attack":
            if action < 36:
                if action in self.playerB and self.checkValidForFight(action):
                    self.fight_attack.append(action)
                    self.playerB.remove(action)
                    self.status = "playerA_defense"
                    return "continue"
                else:
                    return "bad_move"
            if action == 36:
                if not self.fight_attack:
                    return "bad_move"
                self.dump.extend(self.fight_attack)
                self.dump.extend(self.fight_defense)
                self.fight_attack = []
                self.fight_defense = []
                self.drawPreferB()
                if len(self.playerB) == 0 and len(self.playerA) == 0:
                    return "stalemate"
                if len(self.playerB) == 0:
                    return "win_A"
                if len(self.playerA) == 0:
                    return "win_B"
                self.status = "playerA_attack"
                return "continue"
            raise Exception("This exception should never be raised.")

        if self.status == "playerA_attack_add":
            if action < 36:
                if action in self.playerA and self.checkValidForFight(action):
                    self.fight_attack.append(action)
                    self.playerA.remove(action)
                else:
                    return "bad_move"
            elif action == 36:
                for card_id in self.fight_attack + self.fight_defense:
                    self.playerB.append(card_id)
                self.fight_attack = []
                self.fight_defense = []
                self.drawPreferA()
                self.status = "playerA_attack"
                return "continue"

        if self.status == "playerB_attack_add":
            if action < 36:
                if action in self.playerB and self.checkValidForFight(action):
                    self.fight_attack.append(action)
                    self.playerB.remove(action)
                else:
                    return "bad_move"
            elif action == 36:
                for card_id in self.fight_attack + self.fight_defense:
                    self.playerA.append(card_id)
                self.fight_attack = []
                self.fight_defense = []
                self.drawPreferB()
                self.status = "playerB_attack"
                return "continue"

        if self.status == "playerA_defense":
            if action < 36:
                if action in self.playerA:
                    en_r, en_s = get_rs(self.fight_attack[-1])
                    r, s = get_rs(action)
                    if s == en_s and r <= en_r:
                        return "bad_move"
                    if s != en_s and s != self.trump:
                        return "bad_move"
                    self.fight_defense.append(action)
                    self.playerA.remove(action)
                    self.status = "playerB_attack"
                    return "continue"
                else:
                    return "bad_move"
            if action == 36:
                present = False
                for card_id in self.playerB:
                    if self.checkValidForFight(card_id):
                        present = True
                        break
                if not present:
                    for card_id in self.fight_attack + self.fight_defense:
                        self.playerA.append(card_id)
                    self.fight_attack = []
                    self.fight_defense = []
                    self.drawPreferB()
                    self.status = "playerB_attack"
                    return "continue"
                else:
                    self.status = "playerB_attack_add"
                    return "continue"

        if self.status == "playerB_defense":
            if action < 36:
                if action in self.playerB:
                    en_r, en_s = get_rs(self.fight_attack[-1])
                    r, s = get_rs(action)
                    if s == en_s and r <= en_r:
                        return "bad_move"
                    if s != en_s and s != self.trump:
                        return "bad_move"
                    self.fight_defense.append(action)
                    self.playerB.remove(action)
                    self.status = "playerA_attack"
                    return "continue"
                else:
                    return "bad_move"
            if action == 36:
                present = False
                for card_id in self.playerA:
                    if self.checkValidForFight(card_id):
                        present = True
                        break
                if not present:
                    for card_id in self.fight_attack + self.fight_defense:
                        self.playerB.append(card_id)
                    self.fight_attack = []
                    self.fight_defense = []
                    self.drawPreferA()
                    self.status = "playerA_attack"
                    return "continue"
                else:
                    self.status = "playerA_attack_add"
                    return "continue"

    def life_iteration(self, action):
        self.result = self._act(action)
        if self.result == "bad_move":
            print("###", self.status, action, self.get_valid_moves())
            self.observe(2, verbose=True)
        if self.result == "bad_move":
            if "A" in self.status:
                return 0, 0
            else:
                return 0, 1
        else:
            if not self.playerA and not self.playerB:
                self.result = "stalemate"
                return 1, None
            elif not self.playerA:
                self.result = "win_A"
                return 1, None
            elif not self.playerB:
                self.result = "win_B"
                return 1, None
            if "A" in self.status:
                return 0, 0
            else:
                return 0, 1

    def isGameEnd(self):
        if self.result in ["win_A", "win_B", "stalemate"]:
            return True
        else:
            return False

    def isWon(self, player):
        if player == 0 and self.result == "win_A":
            return True
        if player == 1 and self.result == "win_B":
            return True
        return False
    
    def life(self):
        result = "continue"
        while result == "continue" or result == "bad_move":
            if result == "bad_move":
                print(result)
            if "A" in self.status:
                self.observe("A")
                action = int(input("Your action: "))
            elif "B" in self.status:
                action = self.naiveResolve()
            result = self._act(action)
            if result == "continue":
                if not self.playerA and not self.playerB:
                    resut = "stalemate"
                elif not self.playerA:
                    result = "win_A"
                elif not self.playerB:
                    result = "win_B"
        print(result)

    def get_player(self):
        if "A" in self.status:
            player = 0
        else:
            player = 1
        return player

    def observe(self, player, verbose=False):
        observation = np.zeros((333, ))
        if player == -1:
            if "A" in self.status:
                player = 0
            else:
                player = 1
        if player == 0 or player == 2:
            you = self.playerA
            enemy = self.playerB
        elif player == 1:
            you = self.playerB
            enemy = self.playerA
        else:
            raise ValueError("`player` should have value -1 or 0 or 1 or 2.")
        if verbose:
            print("* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * ")
        observation[self.trump_id] = 1
        deck_size = len(self.deck)
        observation[224 + deck_size] = 1
        if verbose:
            r, s = get_rs(self.trump_id)
            r_name = rank_names[r]
            s_name = suit_names[s]
            print("trump: {0} {1}.".format(r_name, s_name))
            print("Deck size: {0}".format(deck_size))
            print("Your cards:")
        for card in you:
            observation[36 + card] = 1
            if verbose:
                r, s = get_rs(card)
                r_name = rank_names[r]
                s_name = suit_names[s]
                print("id: {0}, {1} {2}.".format(card, r_name, s_name))
        if verbose:
            print("Enemy holds {0} cards.".format(len(enemy)))
        if len(enemy) <= 6:
            observation[72 + len(enemy)] = 1
        else:
            observation[72 + 7] = 1
        if player == 2 and verbose:
            print("Enemy cards:")
            for card in enemy:
                r, s = get_rs(card)
                r_name = rank_names[r]
                s_name = suit_names[s]
                print("id: {0}, {1} {2}.".format(card, r_name, s_name))
        if "defense" in self.status:
            if verbose:
                print("Attack cards:")
            for card in self.fight_attack:
                observation[80 + card] = 1
                if verbose:
                    r, s = get_rs(card)
                    r_name = rank_names[r]
                    s_name = suit_names[s]
                    print("id: {0}, {1} {2}.".format(card, r_name, s_name))
            if verbose:
                print("Defense cards:")
            for card in self.fight_defense:
                observation[116 + card] = 1
                if verbose:
                    r, s = get_rs(card)
                    r_name = rank_names[r]
                    s_name = suit_names[s]
                    print("id: {0}, {1} {2}.".format(card, r_name, s_name))
        elif "attack" in self.status:
            if verbose:
                print("Attack cards:")
            for card in self.fight_attack:
                observation[152 + card] = 1
                if verbose:
                    r, s = get_rs(card)
                    r_name = rank_names[r]
                    s_name = suit_names[s]
                    print("id: {0}, {1} {2}.".format(card, r_name, s_name))
            if verbose:
                print("Defense cards:")
            for card in self.fight_defense:
                observation[188 + card] = 1
                if verbose:
                    r, s = get_rs(card)
                    r_name = rank_names[r]
                    s_name = suit_names[s]
                    print("id: {0}, {1} {2}.".format(card, r_name, s_name))
        else:
            raise("Unknown status: {0}".format(self.status))
        if verbose:
            print("Dumped cards:")
        if self.fight_attack:
            observation[260 + self.fight_attack[-1]] = 1
        for card in self.dump:
            if verbose:
                r, s = get_rs(card)
                r_name = rank_names[r]
                s_name = suit_names[s]
                print("id: {0}, {1} {2}.".format(card, r_name, s_name))
            observation[296 + card] = 1
        if verbose:
            print("Status: {0}".format(self.status))
        if "attack" in self.status:
            observation[332] = 1
        observation = observation.reshape((333, 1))
        return observation

    def naiveResolve(self):
        if len(self.playerB) == 0:
            return 36
        if self.status == "playerB_attack" or self.status == "playerB_attack_add":
            cards = [get_rs(card) for card in self.playerB]
            if not self.fight_attack:
                regular_cards = list(filter(lambda rs: rs[1] != self.trump, cards))
                sorted_regular_cards = sorted(regular_cards, key=lambda rs: rs[0])
                if len(sorted_regular_cards) > 0:
                    return get_id(*sorted_regular_cards[0])
                else:
                    return sorted(self.playerB)[0]
            else:
                for card in self.playerB:
                    if self.checkValidForFight(card):
                        return card
                return 36
        if self.status == "playerB_defense":
            en_r, en_s = get_rs(self.fight_attack[-1])
            for card in self.playerB:
                r, s = get_rs(card)
                if s == en_s and r > en_r:
                    return card
            for card in self.playerB:
                r, s = get_rs(card)
                if s != en_s and s == self.trump:
                    return card
            return 36
        return 36

    def deepcopy(self):
        new_world = WorldDurak()
        new_world.status = self.status
        new_world.deck = self.deck.copy()
        new_world.trump = self.trump
        new_world.trump_id = self.trump_id
        new_world.playerA = self.playerA.copy()
        new_world.playerB = self.playerB.copy()
        new_world.fight_attack = self.fight_attack.copy()
        new_world.fight_defense = self.fight_defense.copy()
        new_world.dump = self.dump.copy()
        new_world.result = self.result
        return new_world

    def reset_hidden_state(self, preserved_player):
        if preserved_player == 0:
            this_player = self.playerA
            enemy_player = self.playerB
        elif preserved_player == 1:
            this_player = self.playerB
            enemy_player = self.playerA
        else:
            raise ValueError("Unknown player: {0}. Should be 0 or 1.".format(preserved_player))
        deck_size = len(self.deck)
        if not deck_size:
            return
        unknown_cards = self.deck + enemy_player
        np.random.shuffle(unknown_cards)
        self.deck = unknown_cards[:deck_size]
        enemy_player = unknown_cards[deck_size:]

if __name__ is "__main__":
    world = WorldDurak()
    world.life()
