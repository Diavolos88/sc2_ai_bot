#
# async def attack(self):
#     if len(self.units(UnitTypeId.VOIDRAY).idle) > 0:
#         if self.use_model:
#             prediction = self.model.predict([self.flipped.reshape([-1, 184, 176, 3])])
#             choice = np.argmax(prediction[0])
#             choice_dict = {0: "No Attack!",
#                            1: "Attack close to our nexus!",
#                            2: "Attack Enemy Structure!",
#                            3: "Attack Eneemy Start!"}
#             print("Choice #{}:{}".format(choice, choice_dict[choice]))
#         else:
#             choice = random.randrange(0, 4)
#         target = False
#         if self.gameTime > self.do_something_after:
#             if choice == 0:
#                 # no attack
#                 wait = random.randrange(7, 100) / 100
#                 self.do_something_after = self.gameTime + wait
#             elif choice == 1:
#                 # attack_unit_closest_nexus
#                 if len(self.known_enemy_units) > 0:
#                     target = self.known_enemy_units.closest_to(random.choice(self.units(UnitTypeId.NEXUS)))
#             elif choice == 2:
#                 # attack enemy structures
#                 if len(self.known_enemy_structures) > 0:
#                     target = random.choice(self.known_enemy_structures)
#             elif choice == 3:
#                 # attack_enemy_start
#                 target = self.enemy_start_locations[0]
#             if target:
#                 for vr in self.units(UnitTypeId.VOIDRAY).idle:
#                     await self.do(vr.attack(target))
#             y = np.zeros(4)
#             y[choice] = 1
#             # print(y)
#             self.train_data.append(np.array([y, self.flipped], dtype=object))
#
#
#     async def offensive_force_buildings(self):
#         if self.units(UnitTypeId.PYLON).ready.exists:
#             pylon = self.units(UnitTypeId.PYLON).ready.random
#             if self.units(UnitTypeId.GATEWAY).ready.exists:
#                 if not self.units(UnitTypeId.CYBERNETICSCORE):
#                     if self.can_afford(UnitTypeId.CYBERNETICSCORE) and not self.already_pending(
#                             UnitTypeId.CYBERNETICSCORE):
#                         await self.build(UnitTypeId.CYBERNETICSCORE, near=pylon)
#
#                 # elif self.units(UnitTypeId.GATEWAY).amount < self.gameTime:
#                 elif self.units(UnitTypeId.GATEWAY).amount < 1:
#                     if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
#                         await self.build(UnitTypeId.GATEWAY, near=pylon)
#                 elif self.units(UnitTypeId.STARGATE).amount < self.gameTime:
#                     if self.can_afford(UnitTypeId.STARGATE) and not self.already_pending(UnitTypeId.STARGATE):
#                         await self.build(UnitTypeId.STARGATE, near=pylon)
#                 if self.units(UnitTypeId.CYBERNETICSCORE).ready.exists:
#                     if self.units(UnitTypeId.ROBOTICSFACILITY).amount < 1:
#                         if self.can_afford(UnitTypeId.ROBOTICSFACILITY) and not self.already_pending(
#                                 UnitTypeId.ROBOTICSFACILITY):
#                             await self.build(UnitTypeId.ROBOTICSFACILITY, near=pylon)
#             else:
#                 if not self.units(UnitTypeId.GATEWAY):
#                     if self.can_afford(UnitTypeId.GATEWAY) and not self.already_pending(UnitTypeId.GATEWAY):
#                         await self.build(UnitTypeId.GATEWAY, near=pylon)
#
#     async def build_offensive_force(self):
#         # for gw in self.units(UnitTypeId.GATEWAY).ready.noqueue:
#         #     if self.units(UnitTypeId.STALKER).amount < (self.units(UnitTypeId.VOIDRAY).amount + 1) * 2:
#         #         if self.can_afford(UnitTypeId.STALKER) and self.supply_left > 0:
#         #             await self.do(gw.train(UnitTypeId.STALKER))
#         for sg in self.units(UnitTypeId.STARGATE).ready.noqueue:
#             if self.can_afford(UnitTypeId.VOIDRAY) and self.supply_left > 0:
#                 await self.do(sg.train(UnitTypeId.VOIDRAY))
