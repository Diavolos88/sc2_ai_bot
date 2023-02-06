import sc2
from sc2.ids.unit_typeid import UnitTypeId
import cv2
import keras  # Keras 2.1.2 and TF-GPU 1.9.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import random
from sc2 import run_game, maps, Race, Difficulty, position, Result
import time
import math
import random
import config as cfg



HEADLESS = False

class DiavolosBot(sc2.BotAI):

    def __init__(self, use_model=False, title=1):
        self.title = title
        self.gameTime = 0
        self.MAX_WORKERS = 66
        self.do_something_after = 0
        self.train_data = []
        self.use_model = use_model
        self.scouts_and_spots = {}
        self.choices = {0: self.build_scout,
                        1: self.build_zealot,
                        2: self.build_gateway,
                        3: self.build_voidray,
                        4: self.build_stalker,
                        5: self.build_worker,
                        6: self.build_assimilator,
                        7: self.build_stargate,
                        8: self.build_pylon,
                        9: self.defend_nexus,
                        10: self.attack_known_enemy_unit,
                        11: self.attack_known_enemy_structure,
                        12: self.expand,
                        13: self.do_nothing,
                        14: self.attack_enemy_start_base
                        }

        if self.use_model:
           print("USING MODEL!")
           self.model = keras.models.load_model(cfg.MODEL_PATH)



    async def intel(self):
        draw_dict = {
            UnitTypeId.NEXUS: [15, (0, 255, 0)],
            UnitTypeId.PYLON: [3, (20, 235, 0)],
            UnitTypeId.PROBE: [1, (55, 200, 0)],
            UnitTypeId.ASSIMILATOR: [2, (55, 200, 0)],
            UnitTypeId.ROBOTICSFACILITY: [5, (215, 155, 0)],
            UnitTypeId.GATEWAY: [3, (200, 100, 0)],
            UnitTypeId.CYBERNETICSCORE: [3, (150, 150, 0)],
            UnitTypeId.STARGATE: [5, (255, 0, 0)],
            UnitTypeId.VOIDRAY: [3, (255, 100, 0)]
        }
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0
        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0
        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0
        plausible_supply = self.supply_cap / 200.0
        military_weight = len(self.units(UnitTypeId.VOIDRAY)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0
        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),
                 3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),
                 3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        for unit_type in draw_dict:
            for nexus in self.units(unit_type):
                pos = nexus.position
                # print(pos)
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), draw_dict[unit_type][0], draw_dict[unit_type][1], -1)

        main_base_names = ['nexus', 'commandcenter', 'orbitalcommand', 'planetaryfortress', 'hatchery']
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_base_names:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for obs in self.units(UnitTypeId.OBSERVER):
            pos = obs.position
            # print(pos)
            cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (255, 255, 255), -1)

        for enemy_unit in self.known_enemy_units:
            if not enemy_unit.is_structure:
                worker_names = ["probe",
                                "scv",
                                "drone"]
                # if that unit is a PROBE, SCV, or DRONE... it's a worker
                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        self.flipped = cv2.flip(game_data, 0)

        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow(str(self.title), resized)
            cv2.waitKey(1)

    async def build_scout(self):
        if len(self.units(UnitTypeId.OBSERVER)) < math.floor(self.gameTime / 3):
            for rf in self.units(UnitTypeId.ROBOTICSFACILITY).ready.noqueue:
                # print(len(self.units(UnitTypeId.OBSERVER)), self.gameTime / 3)
                if self.can_afford(UnitTypeId.OBSERVER) and self.supply_left > 0:
                    await self.do(rf.train(UnitTypeId.OBSERVER))



    def on_end(self, game_result):
        print('--- on_end called ---')
        print(game_result, self.use_model)
        if game_result == Result.Victory:
            np.save(cfg.TRAIN_DATA_PATH.format(str(int(time.time()))), np.array(self.train_data))

        with open(cfg.RESULTS_PATH,"a") as f:
            if self.use_model:
                f.write("Model {}\n".format(game_result))
            else:
                f.write("Random {}\n".format(game_result))

        # print(np.array(self.train_data))

    async def build_worker(self):
        nexuses = self.units(UnitTypeId.NEXUS).ready
        if nexuses.exists:
            if self.can_afford(UnitTypeId.PROBE):
                await self.do(random.choice(nexuses).train(UnitTypeId.PROBE))

    async def build_stargate(self):
        if self.units(UnitTypeId.PYLON).ready.exists:
            pylon = self.units(UnitTypeId.PYLON).ready.random
            if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
                if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
                    await self.build(UnitTypeId.STARGATE, near=pylon)

    async def build_pylon(self):
        nexuses = self.units(UnitTypeId.NEXUS).ready
        if nexuses.exists:
            if self.can_afford(UnitTypeId.PYLON):
                await self.build(UnitTypeId.PYLON, near=self.units(UnitTypeId.NEXUS).first.position.towards(self.game_info.map_center, 5))

    async def expand(self):
        try:
            if self.can_afford(UnitTypeId.NEXUS):
                await self.expand_now()
        except Exception as e:
            print(str(e))

    async def build_assimilator(self):
        for nexus in self.units(UnitTypeId.NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(UnitTypeId.ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(UnitTypeId.ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(UnitTypeId.ASSIMILATOR, vaspene))

    async def scout(self):
        self.expand_dis_dir = {}
        for el in self.expansion_locations:
            distance_to_enemy_start = el.distance_to(self.enemy_start_locations[0])
            self.expand_dis_dir[distance_to_enemy_start] = el
            self.ordered_exp_distances = sorted(k for k in self.expand_dis_dir)

            existing_ids = [unit.tag for unit in self.units]
            to_be_removed = []
            for noted_scout in self.scouts_and_spots:
                if noted_scout not in existing_ids:
                    to_be_removed.append(noted_scout)
            for scout in to_be_removed:
                del self.scouts_and_spots[scout]

            if len(self.units(UnitTypeId.ROBOTICSFACILITY).ready) == 0:
                unit_type = UnitTypeId.PROBE
                unit_limit = 1
            else:
                unit_type = UnitTypeId.OBSERVER
                unit_limit = 15

            assign_scout = True
            if unit_type == UnitTypeId.PROBE:
                for unit in self.units(UnitTypeId.PROBE):
                    if unit.tag in self.scouts_and_spots:
                        assign_scout = False

            if assign_scout:
                if len(self.units(unit_type).idle) > 0:
                    for obs in self.units(unit_type).idle[:unit_limit]:
                        if obs.tag not in self.scouts_and_spots:
                            for dist in self.ordered_exp_distances:
                                try:
                                    location = next(value for key, value in self.expand_dis_dir.items() if key == dist)
                                    # DICT {UNIT_ID:LOCATION}
                                    active_locations = [self.scouts_and_spots[k] for k in self.scouts_and_spots]
                                    if location not in active_locations:
                                        if unit_type == UnitTypeId.PROBE:
                                            for unit in self.units(UnitTypeId.PROBE):
                                                if unit.tag in self.scouts_and_spots:
                                                    continue
                                        await self.do(obs.move(location))
                                        self.scouts_and_spots[obs.tag] = location
                                        break
                                except Exception as e:
                                    pass

                for obs in self.units(unit_type):
                    if obs.tag in self.scouts_and_spots:
                        if obs in [probe for probe in self.units(UnitTypeId.PROBE)]:
                            await self.do(obs.move(self.random_location_variance(self.scouts_and_spots[obs.tag])))


    def finde_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    def random_location_variance(self, location):
        x = location[0]
        y = location[1]
        #  FIXED THIS
        x += random.randrange(-5, 5)
        y += random.randrange(-5, 5)
        if x < 0:
            # print("x below")
            x = 0
        if y < 0:
            # print("y below")
            y = 0
        if x > self.game_info.map_size[0]:
            # print("x above")
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            # print("y above")
            y = self.game_info.map_size[1]
        go_to = position.Point2(position.Pointlike((x, y)))
        return go_to

    async def do_something(self):
        if self.time > self.do_something_after:
            if self.use_model:
                prediction = self.model.predict([self.flipped.reshape([-1, 184, 176, 3])])
                choice = np.argmax(prediction[0])
            else:
                choice = random.randrange(0, len(self.choices))
            try:
                await self.choices[choice]()
            except Exception as e:
                print(str(e))
            y = np.zeros(len(self.choices))
            y[choice] = 1
            self.train_data.append([y, self.flipped])

    async def build_zealot(self):
        gateways = self.units(UnitTypeId.GATEWAY).ready
        if gateways.exists:
            if self.can_afford(UnitTypeId.ZEALOT):
                await self.do(random.choice(gateways).train(UnitTypeId.ZEALOT))

    async def build_gateway(self):
        pylon = self.units(UnitTypeId.PYLON).ready.random
        if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
            await self.build(UnitTypeId.GATEWAY, near=pylon)


    async def build_voidray(self):
        stargates = self.units(UnitTypeId.STARGATE).ready
        if stargates.exists:
            if self.can_afford(UnitTypeId.VOIDRAY):
                await self.do(random.choice(stargates).train(UnitTypeId.VOIDRAY))

    async def build_stalker(self):
        pylon = self.units(UnitTypeId.PYLON).ready.random
        gateways = self.units(UnitTypeId.GATEWAY).ready
        cybernetics_cores = self.units(UnitTypeId.CYBERNETICSCORE).ready
        if gateways.exists and cybernetics_cores.exists:
            if self.can_afford(UnitTypeId.STALKER):
                await self.do(random.choice(gateways).train(UnitTypeId.STALKER))
        if not cybernetics_cores.exists:
            if self.units(UnitTypeId.GATEWAY).ready.exists:
                if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(UnitTypeId.CYBERNETICSCORE):
                    await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)





    async def do_nothing(self):
        wait = random.randrange(7, 100)/100
        self.do_something_after = self.time + wait

    async def defend_nexus(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))
            for u in self.units(UnitTypeId.VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.ZEALOT).idle:
                await self.do(u.attack(target))


    async def attack_known_enemy_structure(self):
        if len(self.known_enemy_structures) > 0:
            target = random.choice(self.known_enemy_structures)
            for u in self.units(UnitTypeId.VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.ZEALOT).idle:
                await self.do(u.attack(target))


    async def attack_known_enemy_unit(self):
        if len(self.known_enemy_units) > 0:
            target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))
            for u in self.units(UnitTypeId.VOIDRAY).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.STALKER).idle:
                await self.do(u.attack(target))
            for u in self.units(UnitTypeId.ZEALOT).idle:
                await self.do(u.attack(target))

    async def attack_enemy_start_base(self):
        if len(self.known_enemy_units) > 0:
            target = self.enemy_start_locations[0]
            if target:
                for u in self.units(UnitTypeId.VOIDRAY).idle:
                    await self.do(u.attack(target))
                for u in self.units(UnitTypeId.STALKER).idle:
                    await self.do(u.attack(target))
                for u in self.units(UnitTypeId.ZEALOT).idle:
                    await self.do(u.attack(target))


    async def on_step(self, iteration):
        self.gameTime = (self.state.game_loop/22.4) / 60

        # what to do every step
        # print('Time:', self.gameTime)

        await self.distribute_workers()
        await self.scout()
        await self.intel()
        await self.do_something()

